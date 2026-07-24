import json
import traceback
from typing import List

import mlx.core as mx
import numpy as np


class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device=None,
        stochastic_latent=False,
        enable_compile=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name="NoName",
        boltzmann_epsilon=0,
        boltzmann_temp=1,
        top_p=1,
    ):
        self.engine_type = "mortal"
        self.device = device or mx.default_device()
        self.brain = brain.eval()
        self.dqn = dqn.eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        self._forward_fn = self._forward
        if enable_compile:
            self._forward_fn = mx.compile(
                self._forward,
                inputs=[self.brain.state, self.dqn.state],
            )

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with mx.stream(self.device):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f"{ex}\n{traceback.format_exc()}") from ex

    def _forward(self, obs, masks, invisible_obs=None):
        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                return mu, logsig
            case 2 | 3 | 4:
                return self.dqn(self.brain(obs, invisible_obs), masks)
            case _:
                raise ValueError(f"Unexpected version {self.version}")

    def _react_batch(self, obs, masks, invisible_obs):
        obs = mx.array(np.stack(obs, axis=0), dtype=mx.float32)
        masks = mx.array(np.stack(masks, axis=0), dtype=mx.bool_)
        if invisible_obs is not None:
            invisible_obs = mx.array(
                np.stack(invisible_obs, axis=0), dtype=mx.float32
            )
        batch_size = obs.shape[0]

        if self.version == 1:
            mu, logsig = self._forward_fn(obs, masks, invisible_obs)
            if self.stochastic_latent:
                latent = mu + (mx.exp(logsig) + 1e-6) * mx.random.normal(
                    mu.shape
                )
            else:
                latent = mu
            q_out = self.dqn(latent, masks)
        else:
            q_out = self._forward_fn(obs, masks, invisible_obs)

        if self.boltzmann_epsilon > 0:
            is_greedy = (
                mx.random.uniform(shape=(batch_size,))
                < 1 - self.boltzmann_epsilon
            )
            logits = mx.where(
                masks, q_out / self.boltzmann_temp, -mx.inf
            )
            sampled = sample_top_p(logits, self.top_p)
            actions = mx.where(is_greedy, mx.argmax(q_out, axis=-1), sampled)
        else:
            is_greedy = mx.ones((batch_size,), dtype=mx.bool_)
            actions = mx.argmax(q_out, axis=-1)

        mx.eval(actions, q_out, masks, is_greedy)
        return (
            actions.tolist(),
            q_out.tolist(),
            masks.tolist(),
            is_greedy.tolist(),
        )


def sample_top_p(logits, p):
    if p >= 1:
        return mx.random.categorical(logits)
    if p <= 0:
        return mx.argmax(logits, axis=-1)

    indices = mx.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = mx.take_along_axis(logits, indices, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    remove = mx.cumsum(sorted_probs, axis=-1) - sorted_probs > p
    filtered_logits = mx.where(remove, -mx.inf, sorted_logits)
    sampled_sorted = mx.random.categorical(filtered_logits)
    return mx.squeeze(
        mx.take_along_axis(indices, sampled_sorted[:, None], axis=-1),
        axis=-1,
    )


class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = "mjai-log"
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]["type"] == "start_kyoku"

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": tile,
                            "tsumogiri": True,
                        }
                    )
                )
            else:
                res.append('{"type":"none"}')
        return res

    def start_game(self, game_idx: int):
        pass

    def end_kyoku(self, game_idx: int):
        pass

    def end_game(self, game_idx: int, scores: List[int]):
        pass
