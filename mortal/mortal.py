import prelude

import json
import os
import sys
from datetime import datetime, timezone

import mlx.core as mx

from backend import configure_device
from checkpoint import load_checkpoint
from common import filtered_trimmed_lines
from config import config
from engine import MortalEngine
from libriichi.dataset import Grp
from libriichi.mjai import Bot
from model import GRP
from policy import load_policy


USAGE = """Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3]."""


def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except (AssertionError, ValueError):
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get("MORTAL_REVIEW_MODE", "0") == "1"

    configure_device(config["control"].get("device", "auto"))
    mortal, dqn, state = load_policy(config["control"]["state_file"])
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    num_blocks = cfg["resnet"]["num_blocks"]
    conv_channels = cfg["resnet"]["conv_channels"]
    if "tag" in state:
        tag = state["tag"]
    else:
        timestamp = datetime.fromtimestamp(
            state["timestamp"], tz=timezone.utc
        ).strftime("%y%m%d%H")
        tag = (
            f"mortal{version}-b{num_blocks}c{conv_channels}-t{timestamp}"
        )

    engine = MortalEngine(
        mortal,
        dqn,
        version=version,
        is_oracle=False,
        enable_compile=config["control"].get("enable_compile", False),
        enable_quick_eval=not review_mode,
        enable_rule_based_agari_guard=True,
        name="mortal",
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []
    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if reaction := bot.react(line):
            print(reaction, flush=True)
        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    if review_mode:
        grp = GRP(**config["grp"]["network"])
        load_checkpoint(
            config["grp"]["state_file"],
            models={"grp": grp},
        )

        instance = Grp.load_log("\n".join(logs))
        feature = instance.take_feature()
        sequences = [
            mx.array(feature[: index + 1], dtype=mx.float32)
            for index in range(len(feature))
        ]
        matrix = grp.calc_matrix(grp(sequences))
        mx.eval(matrix)
        print(
            json.dumps(
                {
                    "model_tag": tag,
                    "phi_matrix": matrix.tolist(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
