import prelude

import logging
import socket
import numpy as np
import time
import gc
from os import path
from backend import clear_cache, configure_device
from checkpoint import load_model_weights
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    configure_device(config['control'].get('device', 'auto'))
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(
        version=version,
        num_blocks=num_blocks,
        conv_channels=conv_channels,
    ).eval()
    dqn = DQN(version=version).eval()

    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        load_model_weights(mortal, rsp['mortal'])
        load_model_weights(dqn, rsp['dqn'])
        logging.info('param has been updated')

        rankings, file_list = train_player.train_play(mortal, dqn)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        clear_cache()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
