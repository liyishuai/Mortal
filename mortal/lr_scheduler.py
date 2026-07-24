import math

import mlx.core as mx


class LinearWarmUpCosineAnnealingLR:
    """MLX learning-rate callable matching Mortal's microbatch schedule."""

    def __init__(
        self,
        *,
        peak,
        final,
        warm_up_steps,
        max_steps,
        init=1e-8,
        offset=0,
        epoch_size=0,
        step_scale=1,
        **_,
    ):
        assert peak >= final >= init >= 0
        assert max_steps >= warm_up_steps
        self.init = init
        self.peak = peak
        self.final = final
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.offset = offset
        self.epoch_size = epoch_size
        self.step_scale = step_scale

    def __call__(self, optimizer_steps):
        steps = (
            optimizer_steps * self.step_scale
            + (self.step_scale - 1)
            + self.offset
        )
        if self.epoch_size > 0:
            steps = mx.remainder(steps, self.epoch_size)

        value = mx.full((), self.final, dtype=mx.float32)
        decay_steps = self.max_steps - self.warm_up_steps
        if decay_steps > 0:
            cosine_step = mx.minimum(
                mx.maximum(steps - self.warm_up_steps, 0), decay_steps
            )
            cosine = 0.5 * (
                1 + mx.cos(cosine_step / decay_steps * math.pi)
            )
            decay_value = self.final + (self.peak - self.final) * cosine
            value = mx.where(steps < self.max_steps, decay_value, value)
        if self.warm_up_steps > 0:
            warm_value = self.init + (
                (self.peak - self.init) * steps / self.warm_up_steps
            )
            value = mx.where(steps < self.warm_up_steps, warm_value, value)
        return value
