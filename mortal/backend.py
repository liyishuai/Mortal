import logging
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map


def configure_device(name: str = "auto"):
    """Select the MLX device used by subsequently created arrays and modules."""
    normalized = name.lower()
    if normalized in {"auto", "default"}:
        device = mx.gpu if mx.metal.is_available() else mx.cpu
    elif normalized in {"gpu", "metal", "mps"}:
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU support is not available on this machine")
        device = mx.gpu
    elif normalized == "cpu":
        device = mx.cpu
    else:
        raise ValueError(
            f"Unsupported MLX device {name!r}; expected 'auto', 'gpu', or 'cpu'"
        )
    mx.set_default_device(device)
    logging.info("MLX device: %s", mx.default_device())
    return device


def clear_cache():
    if mx.metal.is_available():
        mx.metal.clear_cache()


def parameter_count(module) -> int:
    return sum(value.size for _, value in tree_flatten(module.trainable_parameters()))


def tree_add(left: Any, right: Any):
    return tree_map(lambda a, b: a + b, left, right)


def tree_scale(tree: Any, scale: float):
    return tree_map(lambda value: value * scale, tree)


def scalar(value) -> float:
    mx.eval(value)
    return float(value.item())
