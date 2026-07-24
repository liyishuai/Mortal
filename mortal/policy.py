from checkpoint import load_checkpoint, read_checkpoint_state
from model import Brain, DQN


def load_policy(file):
    state = read_checkpoint_state(file)
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    mortal = Brain(
        version=version,
        conv_channels=cfg["resnet"]["conv_channels"],
        num_blocks=cfg["resnet"]["num_blocks"],
    )
    dqn = DQN(version=version)
    load_checkpoint(
        file,
        models={"mortal": mortal, "current_dqn": dqn},
    )
    return mortal, dqn, state
