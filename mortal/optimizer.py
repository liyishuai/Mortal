import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map_with_path


class SelectiveAdamW(optim.Adam):
    """Adam with decoupled weight decay selected by a parameter predicate."""

    def __init__(
        self,
        *,
        learning_rate,
        betas,
        eps,
        weight_decay,
        decay_predicate,
        bias_correction=True,
    ):
        super().__init__(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            bias_correction=bias_correction,
        )
        self._decay_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_predicate = decay_predicate

    def update(self, model, gradients):
        if self.weight_decay:
            learning_rate = (
                self._decay_learning_rate(self.step)
                if callable(self._decay_learning_rate)
                else self._decay_learning_rate
            )
            decayed = tree_map_with_path(
                lambda path, value: value
                * (
                    1
                    - mx.array(learning_rate).astype(value.dtype)
                    * self.weight_decay
                )
                if self.decay_predicate(path, value)
                else value,
                model.trainable_parameters(),
            )
            model.update(decayed)
        super().update(model, gradients)
