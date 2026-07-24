import prelude

import logging
import random
from datetime import datetime
from glob import glob
from os import path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tensorboardX import SummaryWriter

from backend import configure_device
from checkpoint import (
    load_checkpoint,
    load_json,
    save_checkpoint,
    save_json,
)
from common import tqdm
from config import config
from libriichi.dataset import Grp
from model import GRP


class GrpFileDatasetsIter:
    def __init__(self, file_list, file_batch_size=50, cycle=False):
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.cycle = cycle
        self.buffer = []

    def build_iter(self):
        while True:
            random.shuffle(self.file_list)
            for start_idx in range(0, len(self.file_list), self.file_batch_size):
                self.populate_buffer(start_idx)
                buffer_size = len(self.buffer)
                for index in random.sample(range(buffer_size), buffer_size):
                    yield self.buffer[index]
                self.buffer.clear()
            if not self.cycle:
                break

    def populate_buffer(self, start_idx):
        file_list = self.file_list[
            start_idx : start_idx + self.file_batch_size
        ]
        data = Grp.load_gz_log_files(file_list)

        for game in data:
            feature = game.take_feature()
            rank_by_player = game.take_rank_by_player()
            for index in range(feature.shape[0]):
                self.buffer.append(
                    (
                        np.asarray(feature[: index + 1], dtype=np.float32),
                        rank_by_player,
                    )
                )

    def __iter__(self):
        return self.build_iter()


def collate(batch):
    lengths = np.asarray(
        [len(inputs) for inputs, _ in batch], dtype=np.int32
    )
    max_length = int(lengths.max())
    feature_size = batch[0][0].shape[-1]
    padded = np.zeros(
        (len(batch), max_length, feature_size), dtype=np.float32
    )
    rank_by_players = np.empty((len(batch), 4), dtype=np.int32)
    for index, (inputs, rank_by_player) in enumerate(batch):
        padded[index, : len(inputs)] = inputs
        rank_by_players[index] = rank_by_player
    return padded, lengths, rank_by_players


def batched_groups(iterable, batch_size):
    batch = []
    for entry in iterable:
        batch.append(entry)
        if len(batch) == batch_size:
            yield collate(batch)
            batch.clear()


def train():
    cfg = config["grp"]
    batch_size = cfg["control"]["batch_size"]
    save_every = cfg["control"]["save_every"]
    val_steps = cfg["control"]["val_steps"]
    enable_compile = cfg["control"].get("enable_compile", False)
    configure_device(cfg["control"].get("device", "auto"))

    lr = cfg["optim"]["lr"]
    grp = GRP(**cfg["network"])
    optimizer = optim.AdamW(
        learning_rate=lr,
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
        bias_correction=True,
    )
    optimizer.init(grp.trainable_parameters())

    state_file = cfg["state_file"]
    if path.exists(state_file):
        state = load_checkpoint(
            state_file,
            models={"grp": grp},
            optimizer=optimizer,
        )
        timestamp = datetime.fromtimestamp(state["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logging.info("loaded: %s", timestamp)
        if state.get("optimizer_reset", False):
            logging.warning(
                "checkpoint was converted without optimizer state; "
                "the MLX optimizer starts fresh"
            )
        steps = state["steps"]
    else:
        steps = 0
    # Match the original resume behavior: the current config may intentionally
    # override the learning rate stored in the optimizer checkpoint.
    optimizer.learning_rate = lr

    file_index = cfg["dataset"]["file_index"]
    if path.exists(file_index):
        index = load_json(file_index)
        train_file_list = index["train_file_list"]
        val_file_list = index["val_file_list"]
    else:
        logging.info("building file index...")
        train_file_list = []
        val_file_list = []
        for pattern in cfg["dataset"]["train_globs"]:
            train_file_list.extend(glob(pattern, recursive=True))
        for pattern in cfg["dataset"]["val_globs"]:
            val_file_list.extend(glob(pattern, recursive=True))
        train_file_list.sort(reverse=True)
        val_file_list.sort(reverse=True)
        save_json(
            file_index,
            {
                "train_file_list": train_file_list,
                "val_file_list": val_file_list,
            },
        )

    writer = SummaryWriter(cfg["control"]["tensorboard_dir"])
    train_data_loader = iter(
        batched_groups(
            GrpFileDatasetsIter(
                file_list=train_file_list,
                file_batch_size=cfg["dataset"]["file_batch_size"],
                cycle=True,
            ),
            batch_size,
        )
    )
    val_data_loader = iter(
        batched_groups(
            GrpFileDatasetsIter(
                file_list=val_file_list,
                file_batch_size=cfg["dataset"]["file_batch_size"],
                cycle=True,
            ),
            batch_size,
        )
    )

    def loss_fn(inputs, lengths, rank_by_players):
        logits = grp.forward_padded(inputs, lengths)
        labels = grp.get_label(rank_by_players)
        loss = nn.losses.cross_entropy(
            logits, labels, reduction="mean"
        )
        accuracy = mx.mean(
            (mx.argmax(logits, axis=-1) == labels).astype(mx.float32)
        )
        return loss, accuracy

    loss_and_grad = nn.value_and_grad(grp, loss_fn)
    if enable_compile:
        loss_and_grad = mx.compile(
            loss_and_grad,
            inputs=grp.state,
            outputs=grp.state,
        )

    stats = {
        "train_loss": 0.0,
        "train_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
    }
    logging.info("train file list size: %s", f"{len(train_file_list):,}")
    logging.info("val file list size: %s", f"{len(val_file_list):,}")

    approx_percent = (
        steps * batch_size / (len(train_file_list) * 10) * 100
    )
    logging.info(
        "total steps: %s est. %6.3f%%", f"{steps:,}", approx_percent
    )

    progress = tqdm(total=save_every, desc="TRAIN")
    for inputs, lengths, rank_by_players in train_data_loader:
        inputs = mx.array(inputs, dtype=mx.float32)
        lengths = mx.array(lengths, dtype=mx.int32)
        rank_by_players = mx.array(rank_by_players, dtype=mx.int32)
        (loss, accuracy), grads = loss_and_grad(
            inputs, lengths, rank_by_players
        )
        optimizer.update(grp, grads)
        mx.eval(loss, accuracy, grp.parameters(), optimizer.state)
        stats["train_loss"] += float(loss.item())
        stats["train_acc"] += float(accuracy.item())

        steps += 1
        progress.update(1)
        if steps % save_every != 0:
            continue

        progress.close()
        grp.eval()
        validation = tqdm(total=val_steps, desc="VAL")
        for _ in range(val_steps):
            inputs, lengths, rank_by_players = next(val_data_loader)
            inputs = mx.array(inputs, dtype=mx.float32)
            lengths = mx.array(lengths, dtype=mx.int32)
            rank_by_players = mx.array(rank_by_players, dtype=mx.int32)
            logits = grp.forward_padded(inputs, lengths)
            labels = grp.get_label(rank_by_players)
            loss = nn.losses.cross_entropy(
                logits, labels, reduction="mean"
            )
            accuracy = mx.mean(
                (mx.argmax(logits, axis=-1) == labels).astype(mx.float32)
            )
            mx.eval(loss, accuracy)
            stats["val_loss"] += float(loss.item())
            stats["val_acc"] += float(accuracy.item())
            validation.update(1)
        validation.close()
        grp.train()

        writer.add_scalars(
            "loss",
            {
                "train": stats["train_loss"] / save_every,
                "val": stats["val_loss"] / val_steps,
            },
            steps,
        )
        writer.add_scalars(
            "acc",
            {
                "train": stats["train_acc"] / save_every,
                "val": stats["val_acc"] / val_steps,
            },
            steps,
        )
        writer.add_scalar("lr", lr, steps)
        writer.flush()

        for key in stats:
            stats[key] = 0.0
        approx_percent = (
            steps * batch_size / (len(train_file_list) * 10) * 100
        )
        logging.info(
            "total steps: %s est. %6.3f%%", f"{steps:,}", approx_percent
        )
        save_checkpoint(
            state_file,
            models={"grp": grp},
            optimizer=optimizer,
            state={
                "steps": steps,
                "timestamp": datetime.now().timestamp(),
                "config": config,
            },
        )
        progress = tqdm(total=save_every, desc="TRAIN")
    progress.close()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
