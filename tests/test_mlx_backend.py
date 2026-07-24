import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map


ROOT = Path(__file__).resolve().parents[1]
MORTAL = ROOT / "mortal"
sys.path.insert(0, str(MORTAL))
os.environ.setdefault(
    "MORTAL_CFG", str(MORTAL / "config.example.toml")
)

libriichi = types.ModuleType("libriichi")
consts = types.ModuleType("libriichi.consts")
consts.ACTION_SPACE = 46
consts.GRP_SIZE = 7
consts.obs_shape = lambda version: (32, 34)
consts.oracle_obs_shape = lambda version: (4, 34)
sys.modules["libriichi"] = libriichi
sys.modules["libriichi.consts"] = consts

from checkpoint import load_checkpoint, model_weights, save_checkpoint
from common import (
    WireProtocolError,
    pack_msg,
    recv_msg,
    send_msg,
    unpack_msg,
)
from convert_torch_checkpoint import (
    convert_aux,
    convert_brain,
    convert_dqn,
    convert_grp,
)
from engine import MortalEngine, sample_top_p
from model import AuxNet, Brain, DQN, GRP, MortalTrainingModel
from optimizer import SelectiveAdamW


class MLXBackendTests(unittest.TestCase):
    def assert_weights_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key, value in expected.items():
            self.assertTrue(
                bool(mx.allclose(value, actual[key]).item()),
                key,
            )

    def test_brain_versions_and_oracle_shapes(self):
        for version in range(1, 5):
            brain = Brain(
                conv_channels=16, num_blocks=1, version=version
            )
            obs = mx.zeros((2, 32, 34), dtype=mx.float32)
            output = brain(obs)
            if version == 1:
                self.assertEqual(
                    [value.shape for value in output],
                    [(2, 512), (2, 512)],
                )
            else:
                self.assertEqual(output.shape, (2, 1024))

        oracle = Brain(
            conv_channels=16,
            num_blocks=1,
            version=4,
            is_oracle=True,
        )
        output = oracle(
            mx.zeros((2, 32, 34)),
            mx.zeros((2, 4, 34)),
        )
        mx.eval(output)
        self.assertEqual(output.shape, (2, 1024))

    def test_dqn_masks_illegal_actions(self):
        dqn = DQN(version=4)
        mask = mx.array(
            [
                [True, False, True] + [False] * 43,
                [False, True, False] + [False] * 43,
            ]
        )
        q_values = dqn(mx.zeros((2, 1024)), mask)
        mx.eval(q_values)
        self.assertTrue(np.isneginf(np.array(q_values)[~np.array(mask)]).all())
        actions = np.array(mx.argmax(q_values, axis=-1))
        self.assertTrue(np.array(mask)[np.arange(2), actions].all())

    def test_grp_variable_lengths_and_probabilities(self):
        grp = GRP(hidden_size=8, num_layers=2)
        logits = grp(
            [
                mx.zeros((2, 7)),
                mx.ones((5, 7)),
                mx.full((3, 7), 0.5),
            ]
        )
        matrix = grp.calc_matrix(logits)
        labels = grp.get_label(
            mx.array(
                [
                    [0, 1, 2, 3],
                    [3, 2, 1, 0],
                    [1, 3, 0, 2],
                ]
            )
        )
        mx.eval(logits, matrix, labels)
        self.assertEqual(logits.shape, (3, 24))
        self.assertEqual(matrix.shape, (3, 4, 4))
        np.testing.assert_allclose(
            np.array(mx.sum(matrix, axis=-1)),
            np.ones((3, 4)),
            atol=1e-5,
        )
        self.assertEqual(labels.tolist(), [0, 23, 10])

    def test_engine_and_top_p_return_legal_actions(self):
        brain = Brain(conv_channels=16, num_blocks=1, version=4)
        dqn = DQN(version=4)
        engine = MortalEngine(
            brain,
            dqn,
            is_oracle=False,
            version=4,
            enable_compile=True,
            boltzmann_epsilon=1,
            top_p=0.8,
        )
        obs = [np.zeros((32, 34), dtype=np.float32) for _ in range(4)]
        masks = [
            np.array([True, True, False] + [False] * 43)
            for _ in range(4)
        ]
        actions, q_values, returned_masks, is_greedy = engine.react_batch(
            obs, masks, None
        )
        self.assertEqual(len(q_values), 4)
        self.assertEqual(is_greedy, [False] * 4)
        self.assertTrue(
            all(returned_masks[index][action] for index, action in enumerate(actions))
        )

        logits = mx.array([[0.0, 2.0, -mx.inf]])
        self.assertEqual(sample_top_p(logits, 0).tolist(), [1])

        oracle_brain = Brain(
            conv_channels=16,
            num_blocks=1,
            version=4,
            is_oracle=True,
        )
        oracle_engine = MortalEngine(
            oracle_brain,
            DQN(version=4),
            is_oracle=True,
            version=4,
            device=mx.cpu,
        )
        actions, *_ = oracle_engine.react_batch(
            [np.zeros((32, 34), dtype=np.float32)],
            [np.ones((46,), dtype=np.bool_)],
            [np.zeros((4, 34), dtype=np.float32)],
        )
        self.assertEqual(len(actions), 1)

    def test_compiled_training_step_updates_native_modules(self):
        brain = Brain(conv_channels=16, num_blocks=1, version=4)
        dqn = DQN(version=4)
        aux = AuxNet((4,))
        network = MortalTrainingModel(brain, dqn, aux)
        optimizer = SelectiveAdamW(
            learning_rate=1e-3,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.1,
            decay_predicate=lambda key, value: key.endswith(".weight")
            and value.ndim > 1,
        )
        optimizer.init(network.trainable_parameters())

        def loss_fn(obs, labels):
            (logits,) = aux(brain(obs))
            return nn.losses.cross_entropy(
                logits, labels, reduction="mean"
            )

        value_and_grad = mx.compile(
            nn.value_and_grad(network, loss_fn),
            inputs=network.state,
            outputs=network.state,
        )
        loss, grads = value_and_grad(
            mx.zeros((2, 32, 34)), mx.array([0, 1])
        )
        optimizer.update(network, grads)
        mx.eval(loss, network.parameters(), optimizer.state)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertEqual(int(optimizer.step.item()), 1)

    def test_selective_adamw_uses_current_scheduled_rate(self):
        model = nn.Linear(2, 1)
        model.weight = mx.ones_like(model.weight)
        model.bias = mx.ones_like(model.bias)
        optimizer = SelectiveAdamW(
            learning_rate=lambda step: 0.1 * (step + 1),
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=1.0,
            decay_predicate=lambda key, value: key == "weight",
        )
        optimizer.init(model.trainable_parameters())
        zero_grads = tree_map(mx.zeros_like, model.trainable_parameters())
        optimizer.update(model, zero_grads)
        optimizer.update(model, zero_grads)
        mx.eval(model.parameters(), optimizer.state)
        np.testing.assert_allclose(
            np.array(model.weight), np.full((1, 2), 0.72), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(model.bias), np.ones((1,)), atol=1e-6
        )

    def test_checkpoint_and_wire_round_trip(self):
        brain = Brain(conv_channels=16, num_blocks=1, version=4)
        dqn = DQN(version=4)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model.safetensors"
            save_checkpoint(
                file,
                models={"mortal": brain, "current_dqn": dqn},
                state={"steps": 7, "config": {"control": {"version": 4}}},
            )
            restored_brain = Brain(
                conv_channels=16, num_blocks=1, version=4
            )
            restored_dqn = DQN(version=4)
            state = load_checkpoint(
                file,
                models={
                    "mortal": restored_brain,
                    "current_dqn": restored_dqn,
                },
            )
            self.assertEqual(state["steps"], 7)
            for key, value in model_weights(brain).items():
                self.assertTrue(
                    bool(
                        mx.allclose(
                            value, model_weights(restored_brain)[key]
                        ).item()
                    ),
                    key,
                )

        message = {
            "weights": {"x": mx.array([[1.0, 2.0]])},
            "scalar": np.array(3.0, dtype=np.float32),
            "log": b"binary",
        }
        packed = pack_msg(message)
        restored = unpack_msg(packed)
        self.assertEqual(restored["weights"]["x"].tolist(), [[1.0, 2.0]])
        self.assertEqual(restored["scalar"].shape, ())
        self.assertEqual(restored["scalar"].item(), 3.0)
        self.assertEqual(restored["log"], b"binary")
        with self.assertRaisesRegex(
            WireProtocolError, "Upgrade the trainer, server, and workers"
        ):
            unpack_msg(packed[len(b"MORTAL-MLX") + 2 :])

        sender, receiver = socket.socketpair()
        with sender, receiver:
            send_msg(sender, message)
            socket_restored = recv_msg(receiver)
        self.assertEqual(
            socket_restored["weights"]["x"].tolist(), [[1.0, 2.0]]
        )

    def test_checkpoint_optimizer_resume_is_explicit(self):
        brain = Brain(conv_channels=16, num_blocks=1, version=4)
        optimizer = SelectiveAdamW(
            learning_rate=1e-3,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0,
            decay_predicate=lambda _key, _value: False,
        )
        optimizer.init(brain.trainable_parameters())
        optimizer.update(
            brain, tree_map(mx.zeros_like, brain.trainable_parameters())
        )
        mx.eval(brain.parameters(), optimizer.state)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resumable = directory / "resumable.safetensors"
            save_checkpoint(
                resumable,
                models={"mortal": brain},
                optimizer=optimizer,
                state={"steps": 1},
            )
            restored_brain = Brain(
                conv_channels=16, num_blocks=1, version=4
            )
            restored_optimizer = SelectiveAdamW(
                learning_rate=1e-3,
                betas=[0.9, 0.999],
                eps=1e-8,
                weight_decay=0,
                decay_predicate=lambda _key, _value: False,
            )
            restored_optimizer.init(
                restored_brain.trainable_parameters()
            )
            load_checkpoint(
                resumable,
                models={"mortal": restored_brain},
                optimizer=restored_optimizer,
            )
            self.assertEqual(int(restored_optimizer.step.item()), 1)

            weights_only = directory / "weights-only.safetensors"
            save_checkpoint(
                weights_only,
                models={"mortal": brain},
                state={"steps": 1},
            )
            with self.assertRaisesRegex(
                ValueError, "missing optimizer state"
            ):
                load_checkpoint(
                    weights_only,
                    optimizer=restored_optimizer,
                )

            intentional_reset = directory / "reset.safetensors"
            save_checkpoint(
                intentional_reset,
                models={"mortal": brain},
                state={"steps": 1, "optimizer_reset": True},
            )
            load_checkpoint(
                intentional_reset,
                optimizer=restored_optimizer,
            )

    def test_legacy_parameter_layout_conversion(self):
        for version in range(1, 5):
            source_brain = Brain(
                conv_channels=16, num_blocks=1, version=version
            )
            old = {}
            for key, value in model_weights(source_brain).items():
                old_key = key
                old_key = old_key.replace(
                    "latent_net.layers.", "latent_net."
                )
                if key.startswith("encoder.input_conv."):
                    old_key = key.replace(
                        "encoder.input_conv", "encoder.net.0"
                    )
                elif key.startswith("encoder.input_norm."):
                    norm_index = 2 if version != 1 else 1
                    old_key = key.replace(
                        "encoder.input_norm",
                        f"encoder.net.{norm_index}",
                    )
                elif key.startswith("encoder.blocks.0."):
                    block_index = 1 if version != 1 else 3
                    old_key = key.replace(
                        "encoder.blocks.0",
                        f"encoder.net.{block_index}",
                    )
                    old_key = old_key.replace(
                        ".res_unit.layers.", ".res_unit."
                    )
                    old_key = old_key.replace(
                        ".ca.shared_mlp.layers.", ".ca.shared_mlp."
                    )
                elif key.startswith("encoder.output_conv."):
                    old_key = key.replace(
                        "encoder.output_conv", "encoder.net.4"
                    )
                elif key.startswith("encoder.output_linear."):
                    old_key = key.replace(
                        "encoder.output_linear", "encoder.net.7"
                    )
                array = np.array(value)
                if value.ndim == 3 and key.endswith(".weight"):
                    array = array.transpose(0, 2, 1)
                old[old_key] = array

            converted_brain = Brain(
                conv_channels=16, num_blocks=1, version=version
            )
            convert_brain(old, converted_brain, version, 1)
            self.assert_weights_equal(
                model_weights(source_brain),
                model_weights(converted_brain),
            )

            source_dqn = DQN(version=version)
            old_dqn = {
                key.replace(".layers.", "."): np.array(value)
                for key, value in model_weights(source_dqn).items()
            }
            converted_dqn = DQN(version=version)
            convert_dqn(old_dqn, converted_dqn, version)
            self.assert_weights_equal(
                model_weights(source_dqn), model_weights(converted_dqn)
            )

        source_aux = AuxNet((4,))
        converted_aux = AuxNet((4,))
        convert_aux(
            {
                key: np.array(value)
                for key, value in model_weights(source_aux).items()
            },
            converted_aux,
        )
        self.assert_weights_equal(
            model_weights(source_aux), model_weights(converted_aux)
        )

        source_grp = GRP(hidden_size=8, num_layers=2)
        grp_weights = model_weights(source_grp)
        old_grp = {}
        for layer in range(2):
            prefix = f"rnns.{layer}"
            old_grp[f"rnn.weight_ih_l{layer}"] = np.array(
                grp_weights[f"{prefix}.Wx"]
            )
            old_grp[f"rnn.weight_hh_l{layer}"] = np.array(
                grp_weights[f"{prefix}.Wh"]
            )
            bias = np.array(grp_weights[f"{prefix}.b"])
            recurrent_bias = np.array(grp_weights[f"{prefix}.bhn"])
            old_grp[f"rnn.bias_ih_l{layer}"] = bias.copy()
            old_grp[f"rnn.bias_hh_l{layer}"] = np.concatenate(
                (np.zeros(16, dtype=np.float32), recurrent_bias)
            )
        for layer in (0, 2):
            for suffix in ("weight", "bias"):
                old_grp[f"fc.{layer}.{suffix}"] = np.array(
                    grp_weights[f"fc.layers.{layer}.{suffix}"]
                )
        converted_grp = GRP(hidden_size=8, num_layers=2)
        convert_grp(old_grp, converted_grp)
        self.assert_weights_equal(
            model_weights(source_grp), model_weights(converted_grp)
        )

    def test_converted_grp_matches_pytorch_gate_equations(self):
        rng = np.random.default_rng(7)
        batch_size = 2
        sequence_length = 4
        hidden_size = 3
        source = {}
        input_size = 7
        for layer in range(2):
            source[f"rnn.weight_ih_l{layer}"] = (
                rng.normal(size=(3 * hidden_size, input_size)) * 0.1
            ).astype(np.float32)
            source[f"rnn.weight_hh_l{layer}"] = (
                rng.normal(size=(3 * hidden_size, hidden_size)) * 0.1
            ).astype(np.float32)
            source[f"rnn.bias_ih_l{layer}"] = (
                rng.normal(size=(3 * hidden_size,)) * 0.05
            ).astype(np.float32)
            source[f"rnn.bias_hh_l{layer}"] = (
                rng.normal(size=(3 * hidden_size,)) * 0.05
            ).astype(np.float32)
            input_size = hidden_size
        source["fc.0.weight"] = (
            rng.normal(size=(6, 6)) * 0.1
        ).astype(np.float32)
        source["fc.0.bias"] = (
            rng.normal(size=(6,)) * 0.05
        ).astype(np.float32)
        source["fc.2.weight"] = (
            rng.normal(size=(24, 6)) * 0.1
        ).astype(np.float32)
        source["fc.2.bias"] = (
            rng.normal(size=(24,)) * 0.05
        ).astype(np.float32)

        inputs = rng.normal(
            size=(batch_size, sequence_length, 7)
        ).astype(np.float32)
        lengths = np.array([4, 2], dtype=np.int32)
        layer_inputs = inputs
        final_states = []
        for layer in range(2):
            weight_ih = source[f"rnn.weight_ih_l{layer}"]
            weight_hh = source[f"rnn.weight_hh_l{layer}"]
            bias_ih = source[f"rnn.bias_ih_l{layer}"]
            bias_hh = source[f"rnn.bias_hh_l{layer}"]
            hidden = np.zeros(
                (batch_size, hidden_size), dtype=np.float32
            )
            outputs = []
            for time_index in range(sequence_length):
                input_gates = (
                    layer_inputs[:, time_index] @ weight_ih.T + bias_ih
                )
                hidden_gates = hidden @ weight_hh.T + bias_hh
                input_reset, input_update, input_new = np.split(
                    input_gates, 3, axis=-1
                )
                hidden_reset, hidden_update, hidden_new = np.split(
                    hidden_gates, 3, axis=-1
                )
                reset = 1 / (
                    1 + np.exp(-(input_reset + hidden_reset))
                )
                update = 1 / (
                    1 + np.exp(-(input_update + hidden_update))
                )
                new = np.tanh(input_new + reset * hidden_new)
                hidden = (1 - update) * new + update * hidden
                outputs.append(hidden)
            layer_inputs = np.stack(outputs, axis=1)
            final_states.append(
                layer_inputs[np.arange(batch_size), lengths - 1]
            )
        state = np.concatenate(final_states, axis=-1)
        expected = np.maximum(
            state @ source["fc.0.weight"].T + source["fc.0.bias"],
            0,
        )
        expected = (
            expected @ source["fc.2.weight"].T + source["fc.2.bias"]
        )

        converted = GRP(hidden_size=hidden_size, num_layers=2)
        convert_grp(source, converted)
        actual = converted.forward_padded(
            mx.array(inputs), mx.array(lengths)
        )
        mx.eval(actual)
        np.testing.assert_allclose(
            np.array(actual), expected, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
