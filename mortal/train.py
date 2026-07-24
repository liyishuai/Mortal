def train():
    import prelude

    import gc
    import gzip
    import json
    import logging
    import os
    import random
    import shutil
    import sys
    from datetime import datetime
    from glob import glob
    from os import path

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from tensorboardX import SummaryWriter

    from backend import (
        clear_cache,
        configure_device,
        parameter_count,
        scalar,
        tree_add,
        tree_scale,
    )
    from checkpoint import (
        load_checkpoint,
        load_json,
        read_checkpoint_state,
        save_checkpoint,
        save_json,
    )
    from common import drain, filtered_trimmed_lines, submit_param, tqdm
    from config import config
    from dataloader import FileDatasetsIter, batched
    from libriichi.consts import obs_shape
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import AuxNet, Brain, DQN, MortalTrainingModel
    from player import TestPlayer

    version = config["control"]["version"]
    online = config["control"]["online"]
    batch_size = config["control"]["batch_size"]
    opt_step_every = config["control"]["opt_step_every"]
    save_every = config["control"]["save_every"]
    test_every = config["control"]["test_every"]
    submit_every = config["control"]["submit_every"]
    test_games = config["test_play"]["games"]
    min_q_weight = config["cql"]["min_q_weight"]
    next_rank_weight = config["aux"]["next_rank_weight"]
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    configure_device(config["control"].get("device", "auto"))
    enable_compile = config["control"].get("enable_compile", False)

    pts = config["env"]["pts"]
    gamma = config["env"]["gamma"]
    file_batch_size = config["dataset"]["file_batch_size"]
    reserve_ratio = config["dataset"]["reserve_ratio"]
    num_epochs = config["dataset"]["num_epochs"]
    enable_augmentation = config["dataset"]["enable_augmentation"]
    augmented_first = config["dataset"]["augmented_first"]
    eps = config["optim"]["eps"]
    betas = config["optim"]["betas"]
    weight_decay = config["optim"]["weight_decay"]
    max_grad_norm = config["optim"]["max_grad_norm"]
    if "num_workers" in config["dataset"]:
        logging.warning(
            "dataset.num_workers is deprecated and ignored by the native "
            "MLX data pipeline"
        )

    mortal = Brain(version=version, **config["resnet"])
    dqn = DQN(version=version)
    aux_net = AuxNet((4,))
    network = MortalTrainingModel(mortal, dqn, aux_net)

    logging.info("version: %s", version)
    logging.info("obs shape: %s", obs_shape(version))
    logging.info("mortal params: %s", f"{parameter_count(mortal):,}")
    logging.info("dqn params: %s", f"{parameter_count(dqn):,}")
    logging.info("aux params: %s", f"{parameter_count(aux_net):,}")

    mortal.freeze_bn(config["freeze_bn"]["mortal"])

    schedule = LinearWarmUpCosineAnnealingLR(
        step_scale=opt_step_every,
        **config["optim"]["scheduler"],
    )
    from optimizer import SelectiveAdamW

    optimizer = SelectiveAdamW(
        learning_rate=schedule,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        decay_predicate=lambda key, value: key.endswith(".weight")
        and value.ndim > 1,
        bias_correction=True,
    )
    optimizer.init(network.trainable_parameters())

    test_player = TestPlayer()
    best_perf = {"avg_rank": 4.0, "avg_pt": -135.0}
    steps = 0
    state_file = config["control"]["state_file"]
    best_state_file = config["control"]["best_state_file"]
    if path.exists(state_file):
        saved_state = read_checkpoint_state(state_file)
        resume_optimizer = (
            not online
            or saved_state["config"]["control"].get("online", False)
        )
        state = load_checkpoint(
            state_file,
            models={
                "mortal": mortal,
                "current_dqn": dqn,
                "aux_net": aux_net,
            },
            optimizer=optimizer if resume_optimizer else None,
        )
        timestamp = datetime.fromtimestamp(state["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logging.info("loaded: %s", timestamp)
        if state.get("optimizer_reset", False):
            logging.warning(
                "checkpoint was converted without optimizer state; "
                "the MLX optimizer and learning-rate schedule start fresh"
            )
        best_perf = state["best_perf"]
        steps = state["steps"]
        if steps % opt_step_every != 0:
            raise ValueError(
                "Checkpoint step is not aligned with the current "
                "opt_step_every; use a compatible accumulation setting "
                "or start a new optimizer checkpoint"
            )
        if resume_optimizer and not state.get("optimizer_reset", False):
            mx.eval(optimizer.step)
            optimizer_steps = int(optimizer.step.item())
            base_offset = config["optim"]["scheduler"].get("offset", 0)
            schedule.offset = (
                base_offset + steps - optimizer_steps * opt_step_every
            )

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info("param has been submitted")

    writer = SummaryWriter(config["control"]["tensorboard_dir"])
    stats = {"dqn_loss": 0.0, "cql_loss": 0.0, "next_rank_loss": 0.0}
    q_window = []
    q_target_window = []
    grad_accumulator = None
    accumulation_count = 0

    def loss_fn(
        obs,
        actions,
        masks,
        steps_to_done,
        kyoku_rewards,
        player_ranks,
    ):
        q_target_mc = (
            mx.power(
                mx.array(gamma, dtype=mx.float32),
                steps_to_done.astype(mx.float32),
            )
            * kyoku_rewards
        )
        phi = mortal(obs)
        q_out = dqn(phi, masks)
        q = mx.squeeze(
            mx.take_along_axis(q_out, actions[:, None], axis=-1), axis=-1
        )
        dqn_loss = 0.5 * nn.losses.mse_loss(q, q_target_mc)
        cql_loss = mx.array(0.0)
        if not online:
            cql_loss = mx.mean(mx.logsumexp(q_out, axis=-1)) - mx.mean(q)

        (next_rank_logits,) = aux_net(phi)
        next_rank_loss = nn.losses.cross_entropy(
            next_rank_logits,
            player_ranks,
            reduction="mean",
        )
        loss = (
            dqn_loss
            + cql_loss * min_q_weight
            + next_rank_loss * next_rank_weight
        )
        return loss, dqn_loss, cql_loss, next_rank_loss, q, q_target_mc

    loss_and_grad = nn.value_and_grad(network, loss_fn)
    if enable_compile:
        loss_and_grad = mx.compile(
            loss_and_grad,
            inputs=network.state,
            outputs=network.state,
        )

    def checkpoint_state():
        return {
            "steps": steps,
            "timestamp": datetime.now().timestamp(),
            "best_perf": best_perf,
            "config": config,
        }

    def save_state():
        save_checkpoint(
            state_file,
            models={
                "mortal": mortal,
                "current_dqn": dqn,
                "aux_net": aux_net,
            },
            optimizer=optimizer,
            state=checkpoint_state(),
        )

    def write_test_metrics(stat):
        avg_pt = stat.avg_pt([90, 45, 0, -135])
        logging.info("avg rank: %.6f", stat.avg_rank)
        logging.info("avg pt: %.6f", avg_pt)
        writer.add_scalar("test_play/avg_ranking", stat.avg_rank, steps)
        writer.add_scalar("test_play/avg_pt", avg_pt, steps)
        writer.add_scalars(
            "test_play/ranking",
            {
                "1st": stat.rank_1_rate,
                "2nd": stat.rank_2_rate,
                "3rd": stat.rank_3_rate,
                "4th": stat.rank_4_rate,
            },
            steps,
        )
        writer.add_scalars(
            "test_play/behavior",
            {
                "agari": stat.agari_rate,
                "houjuu": stat.houjuu_rate,
                "fuuro": stat.fuuro_rate,
                "riichi": stat.riichi_rate,
            },
            steps,
        )
        writer.add_scalars(
            "test_play/agari_point",
            {
                "overall": stat.avg_point_per_agari,
                "riichi": stat.avg_point_per_riichi_agari,
                "fuuro": stat.avg_point_per_fuuro_agari,
                "dama": stat.avg_point_per_dama_agari,
            },
            steps,
        )
        writer.add_scalar(
            "test_play/houjuu_point", stat.avg_point_per_houjuu, steps
        )
        writer.add_scalar(
            "test_play/point_per_round", stat.avg_point_per_round, steps
        )
        writer.add_scalars(
            "test_play/key_step",
            {
                "agari_jun": stat.avg_agari_jun,
                "houjuu_jun": stat.avg_houjuu_jun,
                "riichi_jun": stat.avg_riichi_jun,
            },
            steps,
        )
        writer.add_scalars(
            "test_play/riichi",
            {
                "agari_after_riichi": stat.agari_rate_after_riichi,
                "houjuu_after_riichi": stat.houjuu_rate_after_riichi,
                "chasing_riichi": stat.chasing_riichi_rate,
                "riichi_chased": stat.riichi_chased_rate,
            },
            steps,
        )
        writer.add_scalar(
            "test_play/riichi_point", stat.avg_riichi_point, steps
        )
        writer.add_scalars(
            "test_play/fuuro",
            {
                "agari_after_fuuro": stat.agari_rate_after_fuuro,
                "houjuu_after_fuuro": stat.houjuu_rate_after_fuuro,
            },
            steps,
        )
        writer.add_scalar("test_play/fuuro_num", stat.avg_fuuro_num, steps)
        writer.add_scalar(
            "test_play/fuuro_point", stat.avg_fuuro_point, steps
        )
        writer.flush()
        return avg_pt

    def train_epoch():
        nonlocal steps
        nonlocal best_perf
        nonlocal q_window
        nonlocal q_target_window
        nonlocal grad_accumulator
        nonlocal accumulation_count

        if online:
            player_names = ["trainee"]
            dirname = drain()
            file_list = [
                path.join(dirname, filename)
                for filename in os.listdir(dirname)
            ]
        else:
            player_names_set = set()
            for filename in config["dataset"]["player_names_files"]:
                with open(filename) as stream:
                    player_names_set.update(filtered_trimmed_lines(stream))
            player_names = list(player_names_set)
            logging.info("loaded %s players", f"{len(player_names):,}")

            file_index = config["dataset"]["file_index"]
            if path.exists(file_index):
                file_list = load_json(file_index)["file_list"]
            else:
                logging.info("building file index...")
                file_list = []
                for pattern in config["dataset"]["globs"]:
                    file_list.extend(glob(pattern, recursive=True))
                if player_names_set:
                    filtered = []
                    for filename in tqdm(file_list, unit="file"):
                        with gzip.open(filename, "rt") as stream:
                            start = json.loads(next(stream))
                            if not set(start["names"]).isdisjoint(
                                player_names_set
                            ):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                save_json(file_index, {"file_list": file_list})
        logging.info("file list size: %s", f"{len(file_list):,}")

        before_next_test = (test_every - steps % test_every) % test_every
        logging.info(
            "total steps: %s (~%s)",
            f"{steps:,}",
            f"{before_next_test:,}",
        )

        random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version=version,
            file_list=file_list,
            pts=pts,
            file_batch_size=file_batch_size,
            reserve_ratio=reserve_ratio,
            player_names=player_names,
            num_epochs=num_epochs,
            enable_augmentation=enable_augmentation,
            augmented_first=augmented_first,
        )
        data_loader = batched(file_data, batch_size, drop_last=True)

        progress = tqdm(
            total=save_every,
            desc="TRAIN",
            initial=steps % save_every,
        )

        for (
            obs,
            actions,
            masks,
            steps_to_done,
            kyoku_rewards,
            player_ranks,
        ) in data_loader:
            obs = mx.array(obs, dtype=mx.float32)
            actions = mx.array(actions, dtype=mx.int32)
            masks = mx.array(masks, dtype=mx.bool_)
            steps_to_done = mx.array(steps_to_done, dtype=mx.int32)
            kyoku_rewards = mx.array(kyoku_rewards, dtype=mx.float32)
            player_ranks = mx.array(player_ranks, dtype=mx.int32)

            legal = mx.take_along_axis(
                masks, actions[:, None], axis=-1
            ).all()
            mx.eval(legal)
            if not bool(legal.item()):
                raise ValueError("Dataset contains an illegal recorded action")

            values, grads = loss_and_grad(
                obs,
                actions,
                masks,
                steps_to_done,
                kyoku_rewards,
                player_ranks,
            )
            (
                _,
                dqn_loss,
                cql_loss,
                next_rank_loss,
                q,
                q_target_mc,
            ) = values
            grads = tree_scale(grads, 1 / opt_step_every)
            grad_accumulator = (
                grads
                if grad_accumulator is None
                else tree_add(grad_accumulator, grads)
            )

            steps += 1
            accumulation_count += 1
            if accumulation_count == opt_step_every:
                if max_grad_norm > 0:
                    grad_accumulator, _ = optim.clip_grad_norm(
                        grad_accumulator, max_grad_norm
                    )
                optimizer.update(network, grad_accumulator)
                grad_accumulator = None
                accumulation_count = 0

            mx.eval(
                values,
                network.parameters(),
                optimizer.state,
            )
            stats["dqn_loss"] += float(dqn_loss.item())
            if not online:
                stats["cql_loss"] += float(cql_loss.item())
            stats["next_rank_loss"] += float(next_rank_loss.item())
            q_window.append(np.array(q))
            q_target_window.append(np.array(q_target_mc))
            progress.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info("param has been submitted")

            if steps % save_every != 0:
                continue

            progress.close()
            all_q = np.concatenate(q_window)[::128]
            all_q_target = np.concatenate(q_target_window)[::128]
            q_window = []
            q_target_window = []

            writer.add_scalar(
                "loss/dqn_loss", stats["dqn_loss"] / save_every, steps
            )
            if not online:
                writer.add_scalar(
                    "loss/cql_loss", stats["cql_loss"] / save_every, steps
                )
            writer.add_scalar(
                "loss/next_rank_loss",
                stats["next_rank_loss"] / save_every,
                steps,
            )
            writer.add_scalar(
                "hparam/lr", scalar(optimizer.learning_rate), steps
            )
            writer.add_histogram("q_predicted", all_q, steps)
            writer.add_histogram("q_target", all_q_target, steps)
            writer.flush()

            for key in stats:
                stats[key] = 0.0

            before_next_test = (test_every - steps % test_every) % test_every
            logging.info(
                "total steps: %s (~%s)",
                f"{steps:,}",
                f"{before_next_test:,}",
            )
            save_state()

            if online and steps % submit_every != 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info("param has been submitted")

            if steps % test_every == 0:
                stat = test_player.test_play(
                    test_games // 4, mortal, dqn
                )
                mortal.train()
                dqn.train()
                avg_pt = write_test_metrics(stat)
                better = (
                    avg_pt >= best_perf["avg_pt"]
                    and stat.avg_rank <= best_perf["avg_rank"]
                )
                if better:
                    past_best = best_perf.copy()
                    best_perf = {
                        "avg_pt": avg_pt,
                        "avg_rank": stat.avg_rank,
                    }
                    save_state()
                    shutil.copy(state_file, best_state_file)
                    logging.info(
                        "new record: pt %.4f -> %.4f, rank %.4f -> %.4f; "
                        "saved to %s",
                        past_best["avg_pt"],
                        best_perf["avg_pt"],
                        past_best["avg_rank"],
                        best_perf["avg_rank"],
                        best_state_file,
                    )
                if online:
                    sys.exit(0)
            progress = tqdm(total=save_every, desc="TRAIN")

        progress.close()
        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info("param has been submitted")

    while True:
        train_epoch()
        gc.collect()
        clear_cache()
        if not online:
            break


def main():
    import os
    import sys
    import time
    from subprocess import Popen

    from config import config

    is_sub_proc_key = "MORTAL_IS_SUB_PROC"
    online = config["control"]["online"]
    if not online or os.environ.get(is_sub_proc_key, "0") == "1":
        train()
        return

    cmd = (sys.executable, __file__)
    env = {is_sub_proc_key: "1", **os.environ.copy()}
    while True:
        child = Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
