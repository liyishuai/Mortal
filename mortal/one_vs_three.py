import prelude

import os
import secrets

import numpy as np

from backend import configure_device
from config import config
from engine import MortalEngine
from libriichi.arena import OneVsThree
from policy import load_policy


def _load_engine(profile):
    mortal, dqn, state = load_policy(profile["state_file"])
    version = state["config"]["control"].get("version", 1)
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        stochastic_latent=profile.get("stochastic_latent", False),
        enable_compile=profile.get("enable_compile", False),
        enable_rule_based_agari_guard=profile[
            "enable_rule_based_agari_guard"
        ],
        name=profile["name"],
    )


def main():
    cfg = config["1v3"]
    games_per_iter = cfg["games_per_iter"]
    seeds_per_iter = games_per_iter // 4
    iters = cfg["iters"]
    log_dir = cfg["log_dir"]
    use_akochan = cfg["akochan"]["enabled"]
    configure_device(cfg["challenger"].get("device", "auto"))

    if (key := cfg.get("seed_key", -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ["AKOCHAN_DIR"] = cfg["akochan"]["dir"]
        os.environ["AKOCHAN_TACTICS"] = cfg["akochan"]["tactics"]
    else:
        engine_cham = _load_engine(cfg["champion"])
    engine_chal = _load_engine(cfg["challenger"])

    seed_start = 10000
    for index, seed in enumerate(
        range(
            seed_start,
            seed_start + seeds_per_iter * iters,
            seeds_per_iter,
        )
    ):
        print("-" * 50)
        print("#", index)
        env = OneVsThree(
            disable_progress_bar=False,
            log_dir=log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine=engine_chal,
                seed_start=(seed, key),
                seed_count=seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger=engine_chal,
                champion=engine_cham,
                seed_start=(seed, key),
                seed_count=seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = (
            rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        )
        print(
            f"challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
