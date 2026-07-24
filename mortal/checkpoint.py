import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


FORMAT_NAME = "mortal-mlx"
SCHEMA_VERSION = 1
MODEL_PREFIX = "model."
OPTIMIZER_PREFIX = "optimizer."


def model_weights(model) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def load_model_weights(model, weights: Mapping[str, mx.array], *, strict=True):
    model.load_weights(
        [
            (
                key,
                value if isinstance(value, mx.array) else mx.array(value),
            )
            for key, value in weights.items()
        ],
        strict=strict,
    )
    mx.eval(model.parameters())
    return model


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _validate_path(file) -> Path:
    file = Path(file)
    if file.suffix != ".safetensors":
        raise ValueError(
            f"MLX checkpoints must use the .safetensors extension: {file}"
        )
    return file


def save_checkpoint(
    file,
    *,
    models: Mapping[str, object],
    optimizer=None,
    state: Optional[dict] = None,
):
    """Atomically save model/optimizer arrays and JSON metadata in Safetensors."""
    file = _validate_path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for model_name, model in models.items():
        for key, value in model_weights(model).items():
            arrays[f"{MODEL_PREFIX}{model_name}.{key}"] = value

    if optimizer is not None:
        for key, value in tree_flatten(optimizer.state):
            arrays[f"{OPTIMIZER_PREFIX}{key}"] = value

    mx.eval(*arrays.values())
    metadata = {
        "format": FORMAT_NAME,
        "schema_version": str(SCHEMA_VERSION),
        "state": json.dumps(
            state or {},
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        ),
    }

    fd, temporary = tempfile.mkstemp(
        dir=file.parent,
        prefix=f".{file.name}.",
        suffix=".safetensors",
    )
    os.close(fd)
    try:
        mx.save_safetensors(temporary, arrays, metadata)
        os.replace(temporary, file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_raw(file):
    file = _validate_path(file)
    arrays, metadata = mx.load(file, return_metadata=True)
    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(f"{file} is not a {FORMAT_NAME} checkpoint")
    schema_version = int(metadata.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    state = json.loads(metadata.get("state", "{}"))
    return arrays, state


def read_checkpoint_state(file) -> dict:
    _, state = _load_raw(file)
    return state


def load_checkpoint(
    file,
    *,
    models: Optional[Mapping[str, object]] = None,
    optimizer=None,
    strict=True,
) -> dict:
    arrays, state = _load_raw(file)

    for model_name, model in (models or {}).items():
        prefix = f"{MODEL_PREFIX}{model_name}."
        weights = {
            key[len(prefix) :]: value
            for key, value in arrays.items()
            if key.startswith(prefix)
        }
        if not weights:
            raise ValueError(f"Checkpoint does not contain model {model_name!r}")
        load_model_weights(model, weights, strict=strict)

    if optimizer is not None:
        optimizer_weights = [
            (key[len(OPTIMIZER_PREFIX) :], value)
            for key, value in arrays.items()
            if key.startswith(OPTIMIZER_PREFIX)
        ]
        if optimizer_weights:
            optimizer.state = tree_unflatten(optimizer_weights)
            mx.eval(optimizer.state)
        elif not state.get("optimizer_reset", False):
            raise ValueError(
                "Checkpoint is missing optimizer state and does not declare "
                "optimizer_reset=true"
            )

    return state


def save_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=file.parent,
        prefix=f".{file.name}.",
        suffix=".json",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(file):
    with open(file, encoding="utf-8") as stream:
        return json.load(stream)
