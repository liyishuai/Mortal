from itertools import permutations
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from libriichi.consts import ACTION_SPACE, GRP_SIZE, obs_shape, oracle_obs_shape


class Identity(nn.Module):
    def __call__(self, x):
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for layer in self.shared_mlp.layers:
                if isinstance(layer, nn.Linear):
                    layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, x):
        # MLX convolutions are NLC, so sequence reduction is over axis 1.
        avg_out = self.shared_mlp(mx.mean(x, axis=1))
        max_out = self.shared_mlp(mx.max(x, axis=1))
        weight = mx.sigmoid(avg_out + max_out)
        return mx.expand_dims(weight, axis=1) * x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        norm_builder=Identity,
        actv_builder=nn.ReLU,
        pre_actv=False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                ),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                ),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                ),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(
            channels, actv_builder=actv_builder, bias=True
        )

    def __call__(self, x):
        out = self.ca(self.res_unit(x)) + x
        if not self.pre_actv:
            out = self.actv(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        *,
        norm_builder=Identity,
        actv_builder=nn.ReLU,
        pre_actv=False,
    ):
        super().__init__()
        self.pre_actv = pre_actv
        self.input_conv = nn.Conv1d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_norm = norm_builder()
        self.input_actv = actv_builder()
        self.blocks = [
            ResBlock(
                conv_channels,
                norm_builder=norm_builder,
                actv_builder=actv_builder,
                pre_actv=pre_actv,
            )
            for _ in range(num_blocks)
        ]
        self.output_conv = nn.Conv1d(
            conv_channels, 32, kernel_size=3, padding=1
        )
        self.output_actv = actv_builder()
        self.output_linear = nn.Linear(32 * 34, 1024)

    def __call__(self, x):
        # libriichi emits PyTorch-compatible NCL observations; MLX Conv1d
        # consumes NLC.
        x = mx.transpose(x, (0, 2, 1))
        x = self.input_conv(x)
        if not self.pre_actv:
            x = self.input_actv(self.input_norm(x))
        for block in self.blocks:
            x = block(x)
        if self.pre_actv:
            x = self.input_actv(self.input_norm(x))
        x = self.output_actv(self.output_conv(x))

        # Preserve the original channel-major flattening so converted Linear
        # weights remain numerically compatible.
        x = mx.transpose(x, (0, 2, 1))
        x = mx.reshape(x, (x.shape[0], -1))
        return self.output_linear(x)


class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        norm_builder = lambda: nn.BatchNorm(
            conv_channels, momentum=0.01
        )
        actv_builder = nn.Mish
        pre_actv = True

        match version:
            case 1:
                actv_builder = nn.ReLU
                pre_actv = False
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                )
                self.mu_head = nn.Linear(512, 512)
                self.logsig_head = nn.Linear(512, 512)
            case 2:
                pass
            case 3 | 4:
                norm_builder = lambda: nn.BatchNorm(
                    conv_channels, momentum=0.01, eps=1e-3
                )
            case _:
                raise ValueError(f"Unexpected version {self.version}")

        self.encoder = ResNet(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            actv_builder=actv_builder,
            pre_actv=pre_actv,
        )
        self.actv = actv_builder()
        self._freeze_bn = False

    def __call__(self, obs, invisible_obs: Optional[mx.array] = None):
        if self.is_oracle:
            if invisible_obs is None:
                raise ValueError("Oracle Brain requires invisible observations")
            obs = mx.concatenate((obs, invisible_obs), axis=1)
        phi = self.encoder(obs)

        match self.version:
            case 1:
                latent_out = self.latent_net(phi)
                return self.mu_head(latent_out), self.logsig_head(latent_out)
            case 2 | 3 | 4:
                return self.actv(phi)
            case _:
                raise ValueError(f"Unexpected version {self.version}")

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_bn:
            self.apply_to_modules(
                lambda _, module: module.train(False)
                if isinstance(module, nn.BatchNorm)
                else None
            )
        return self

    def reset_running_stats(self):
        def reset(_, module):
            if isinstance(module, nn.BatchNorm) and module.track_running_stats:
                module.running_mean = mx.zeros_like(module.running_mean)
                module.running_var = mx.ones_like(module.running_var)

        self.apply_to_modules(reset)
        return self

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)


class AuxNet(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = tuple(dims)
        self.net = nn.Linear(1024, sum(self.dims), bias=False)

    def __call__(self, x):
        split_points = []
        offset = 0
        for dim in self.dims[:-1]:
            offset += dim
            split_points.append(offset)
        return tuple(mx.split(self.net(x), split_points, axis=-1))


class DQN(nn.Module):
    def __init__(self, *, version=1):
        super().__init__()
        self.version = version
        match version:
            case 1:
                self.v_head = nn.Linear(512, 1)
                self.a_head = nn.Linear(512, ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(),
                    nn.Linear(hidden_size, 1),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                self.net = nn.Linear(1024, 1 + ACTION_SPACE)
                self.net.bias = mx.zeros_like(self.net.bias)
            case _:
                raise ValueError(f"Unexpected version {self.version}")

    def __call__(self, phi, mask):
        if self.version == 4:
            v, a = mx.split(self.net(phi), [1], axis=-1)
        else:
            v = self.v_head(phi)
            a = self.a_head(phi)
        a_sum = mx.sum(mx.where(mask, a, 0.0), axis=-1, keepdims=True)
        mask_sum = mx.sum(mask, axis=-1, keepdims=True)
        a_mean = a_sum / mask_sum
        return mx.where(mask, v + a - a_mean, -mx.inf)


class GRP(nn.Module):
    """Game result predictor implemented with stacked native MLX GRUs."""

    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnns = [
            nn.GRU(
                input_size=GRP_SIZE if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ]
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_size * num_layers, 24),
        )

        # Metal does not support float64. GRP is intentionally float32 so both
        # training and reward calculation stay accelerated on Apple Silicon.
        self.perms = mx.array(list(permutations(range(4))), dtype=mx.int32)
        self.perms_t = mx.transpose(self.perms)
        self.freeze(keys=["perms", "perms_t"], recurse=False)

    def __call__(self, inputs, lengths=None):
        if isinstance(inputs, (list, tuple)):
            lengths = mx.array([value.shape[0] for value in inputs], mx.int32)
            max_length = max(value.shape[0] for value in inputs)
            inputs = mx.stack(
                [
                    mx.pad(
                        mx.array(value, dtype=mx.float32),
                        ((0, max_length - value.shape[0]), (0, 0)),
                    )
                    for value in inputs
                ]
            )
        else:
            inputs = mx.array(inputs, dtype=mx.float32)
            if lengths is None:
                lengths = mx.full(
                    (inputs.shape[0],), inputs.shape[1], dtype=mx.int32
                )
            else:
                lengths = mx.array(lengths, dtype=mx.int32)
        return self.forward_padded(inputs, lengths)

    def forward_padded(self, inputs, lengths):
        outputs = inputs
        states = []
        for rnn in self.rnns:
            outputs = rnn(outputs)
            indices = mx.broadcast_to(
                (lengths - 1)[:, None, None],
                (outputs.shape[0], 1, outputs.shape[-1]),
            )
            states.append(
                mx.squeeze(
                    mx.take_along_axis(outputs, indices, axis=1),
                    axis=1,
                )
            )
        return self.fc(mx.concatenate(states, axis=-1))

    # (N, 24) -> (N, player, rank_prob)
    def calc_matrix(self, logits):
        probs = mx.softmax(logits, axis=-1)
        players = []
        for player in range(4):
            ranks = []
            for rank in range(4):
                ranks.append(
                    mx.sum(
                        mx.where(self.perms_t[player] == rank, probs, 0.0),
                        axis=-1,
                    )
                )
            players.append(mx.stack(ranks, axis=-1))
        return mx.stack(players, axis=1)

    # (N, 4) -> (N)
    def get_label(self, rank_by_player):
        rank_by_player = mx.array(rank_by_player, dtype=mx.int32)
        matches = mx.all(
            self.perms[None, :, :] == rank_by_player[:, None, :],
            axis=-1,
        )
        return mx.argmax(matches, axis=-1).astype(mx.int32)


class MortalTrainingModel(nn.Module):
    def __init__(self, mortal, dqn, aux_net):
        super().__init__()
        self.mortal = mortal
        self.dqn = dqn
        self.aux_net = aux_net
