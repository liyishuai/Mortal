"""One-time converter from Mortal PyTorch .pth files to MLX Safetensors."""

import argparse
import logging

import mlx.core as mx
import numpy as np

from checkpoint import load_model_weights, model_weights, save_checkpoint
from model import AuxNet, Brain, DQN, GRP


def _array(tensor, *, float32=False):
    if isinstance(tensor, np.ndarray):
        if float32 and np.issubdtype(tensor.dtype, np.floating):
            tensor = tensor.astype(np.float32)
        return mx.array(tensor)
    tensor = tensor.detach().cpu()
    if float32 and tensor.is_floating_point():
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def _copy_common(source, source_prefix, destination, destination_prefix):
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        source_key = f"{source_prefix}.{suffix}"
        if source_key in source:
            destination[f"{destination_prefix}.{suffix}"] = _array(
                source[source_key]
            )


def convert_brain(source, model, version, num_blocks):
    converted = {}
    pre_actv = version != 1

    _copy_common(source, "encoder.net.0", converted, "encoder.input_conv")
    norm_index = num_blocks + 1 if pre_actv else 1
    _copy_common(
        source,
        f"encoder.net.{norm_index}",
        converted,
        "encoder.input_norm",
    )

    block_offset = 1 if pre_actv else 3
    for block_index in range(num_blocks):
        old_prefix = f"encoder.net.{block_offset + block_index}"
        new_prefix = f"encoder.blocks.{block_index}"
        for key, value in source.items():
            prefix = f"{old_prefix}.res_unit."
            if key.startswith(prefix) and not key.endswith(
                "num_batches_tracked"
            ):
                suffix = key[len(prefix) :]
                layer, name = suffix.split(".", 1)
                converted[
                    f"{new_prefix}.res_unit.layers.{layer}.{name}"
                ] = _array(value)
            prefix = f"{old_prefix}.ca.shared_mlp."
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                layer, name = suffix.split(".", 1)
                converted[
                    f"{new_prefix}.ca.shared_mlp.layers.{layer}.{name}"
                ] = _array(value)

    output_conv_index = (
        num_blocks + 3 if pre_actv else num_blocks + 3
    )
    output_linear_index = (
        num_blocks + 6 if pre_actv else num_blocks + 6
    )
    _copy_common(
        source,
        f"encoder.net.{output_conv_index}",
        converted,
        "encoder.output_conv",
    )
    _copy_common(
        source,
        f"encoder.net.{output_linear_index}",
        converted,
        "encoder.output_linear",
    )

    if version == 1:
        _copy_common(
            source, "latent_net.0", converted, "latent_net.layers.0"
        )
        _copy_common(source, "mu_head", converted, "mu_head")
        _copy_common(source, "logsig_head", converted, "logsig_head")

    for key, value in list(converted.items()):
        if key.endswith(".weight") and value.ndim == 3:
            converted[key] = mx.transpose(value, (0, 2, 1))
    _load_complete(model, converted, "Brain")


def convert_dqn(source, model, version):
    converted = {}
    if version in (1, 4):
        for prefix in ("v_head", "a_head") if version == 1 else ("net",):
            _copy_common(source, prefix, converted, prefix)
    else:
        for head in ("v_head", "a_head"):
            for layer in (0, 2):
                _copy_common(
                    source,
                    f"{head}.{layer}",
                    converted,
                    f"{head}.layers.{layer}",
                )
    _load_complete(model, converted, "DQN")


def convert_aux(source, model):
    converted = {}
    _copy_common(source, "net", converted, "net")
    _load_complete(model, converted, "AuxNet")


def convert_grp(source, model):
    converted = {}
    hidden_size = model.hidden_size
    for layer in range(model.num_layers):
        prefix = f"rnn."
        weight_ih = source[f"{prefix}weight_ih_l{layer}"]
        weight_hh = source[f"{prefix}weight_hh_l{layer}"]
        bias_ih = source[f"{prefix}bias_ih_l{layer}"]
        bias_hh = source[f"{prefix}bias_hh_l{layer}"]
        converted[f"rnns.{layer}.Wx"] = _array(weight_ih, float32=True)
        converted[f"rnns.{layer}.Wh"] = _array(weight_hh, float32=True)
        converted[f"rnns.{layer}.b"] = mx.concatenate(
            (
                _array(
                    bias_ih[: 2 * hidden_size]
                    + bias_hh[: 2 * hidden_size],
                    float32=True,
                ),
                _array(bias_ih[2 * hidden_size :], float32=True),
            )
        )
        converted[f"rnns.{layer}.bhn"] = _array(
            bias_hh[2 * hidden_size :], float32=True
        )

    for layer in (0, 2):
        for suffix in ("weight", "bias"):
            key = f"fc.{layer}.{suffix}"
            converted[f"fc.layers.{layer}.{suffix}"] = _array(
                source[key], float32=True
            )
    defaults = model_weights(model)
    converted["perms"] = defaults["perms"]
    converted["perms_t"] = defaults["perms_t"]
    _load_complete(model, converted, "GRP")


def _load_complete(model, converted, name):
    expected = set(model_weights(model))
    received = set(converted)
    if missing := expected - received:
        raise ValueError(
            f"{name} conversion missed parameters: {sorted(missing)}"
        )
    if extra := received - expected:
        raise ValueError(
            f"{name} conversion produced unknown parameters: {sorted(extra)}"
        )
    load_model_weights(model, converted)


def _metadata(state):
    metadata = {
        key: state[key]
        for key in ("steps", "timestamp", "best_perf", "config", "tag")
        if key in state
    }
    metadata["optimizer_reset"] = True
    return metadata


def convert_policy(state, output):
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    num_blocks = cfg["resnet"]["num_blocks"]
    mortal = Brain(
        version=version,
        conv_channels=cfg["resnet"]["conv_channels"],
        num_blocks=num_blocks,
    )
    dqn = DQN(version=version)
    convert_brain(state["mortal"], mortal, version, num_blocks)
    convert_dqn(state["current_dqn"], dqn, version)
    models = {"mortal": mortal, "current_dqn": dqn}
    if "aux_net" in state:
        aux_net = AuxNet((4,))
        convert_aux(state["aux_net"], aux_net)
        models["aux_net"] = aux_net
    save_checkpoint(output, models=models, state=_metadata(state))


def convert_grp_checkpoint(state, output, hidden_size, num_layers):
    grp = GRP(hidden_size=hidden_size, num_layers=num_layers)
    convert_grp(state["model"], grp)
    save_checkpoint(output, models={"grp": grp}, state=_metadata(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="PyTorch .pth checkpoint")
    parser.add_argument("output", help="MLX .safetensors checkpoint")
    parser.add_argument(
        "--grp",
        action="store_true",
        help="convert a GRP checkpoint instead of a policy checkpoint",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required only for this one-time conversion. "
            "Install it in a temporary environment and rerun the command."
        ) from error

    state = torch.load(args.input, weights_only=True, map_location="cpu")
    if args.grp:
        convert_grp_checkpoint(
            state, args.output, args.hidden_size, args.num_layers
        )
    else:
        convert_policy(state, args.output)
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "converted %s to %s; optimizer state was intentionally reset",
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
