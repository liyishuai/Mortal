import logging
import os
import secrets
import shutil
from os import path

import numpy as np

from config import config
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from policy import load_policy


class TestPlayer:
    def __init__(self):
        baseline_cfg = config["baseline"]["test"]
        stable_mortal, stable_dqn, state = load_policy(
            baseline_cfg["state_file"]
        )
        version = state["config"]["control"].get("version", 1)
        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle=False,
            version=version,
            enable_compile=baseline_cfg.get("enable_compile", False),
            enable_rule_based_agari_guard=True,
            name="baseline",
        )
        self.chal_version = config["control"]["version"]
        self.log_dir = path.abspath(config["test_play"]["log_dir"])

    def test_play(self, seed_count, mortal, dqn):
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle=False,
            version=self.chal_version,
            enable_compile=config["control"].get("enable_compile", False),
            name="mortal",
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar=False,
            log_dir=self.log_dir,
        )
        env.py_vs_py(
            challenger=engine_chal,
            champion=self.baseline_engine,
            seed_start=(10000, 0x2000),
            seed_count=seed_count,
        )
        return Stat.from_dir(self.log_dir, "mortal")


class TrainPlayer:
    def __init__(self):
        baseline_cfg = config["baseline"]["train"]
        stable_mortal, stable_dqn, state = load_policy(
            baseline_cfg["state_file"]
        )
        version = state["config"]["control"].get("version", 1)
        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle=False,
            version=version,
            enable_compile=baseline_cfg.get("enable_compile", False),
            enable_rule_based_agari_guard=True,
            name="baseline",
        )

        profile = os.environ.get("TRAIN_PLAY_PROFILE", "default")
        logging.info("using profile %s", profile)
        cfg = config["train_play"][profile]
        self.chal_version = config["control"]["version"]
        self.log_dir = path.abspath(cfg["log_dir"])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg["games"] // 4
        self.boltzmann_epsilon = cfg["boltzmann_epsilon"]
        self.boltzmann_temp = cfg["boltzmann_temp"]
        self.top_p = cfg["top_p"]
        self.repeats = cfg["repeats"]
        self.repeat_counter = 0

    def train_play(self, mortal, dqn):
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle=False,
            version=self.chal_version,
            boltzmann_epsilon=self.boltzmann_epsilon,
            boltzmann_temp=self.boltzmann_temp,
            top_p=self.top_p,
            enable_compile=config["online"].get("enable_compile", False),
            name="trainee",
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar=False,
            log_dir=self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger=engine_chal,
            champion=self.baseline_engine,
            seed_start=(self.train_seed, self.train_key),
            seed_count=self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = [
            path.join(self.log_dir, filename)
            for filename in os.listdir(self.log_dir)
        ]
        return rankings, file_list
