import mlx.core as mx
import numpy as np


class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False):
        self.grp = grp.eval()
        self.uniform_init = uniform_init
        self.pts = mx.array(pts or [3, 1, -1, -3], dtype=mx.float32)

    def calc_grp(self, grp_feature):
        sequences = [
            mx.array(grp_feature[: idx + 1], dtype=mx.float32)
            for idx in range(len(grp_feature))
        ]
        logits = self.grp(sequences)
        matrix = self.grp.calc_matrix(logits)
        mx.eval(matrix)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = np.zeros((1, 4), dtype=np.float32)
        final_ranking[0, rank_by_player[player_id]] = 1.0
        rank_prob = mx.concatenate(
            (matrix[:, player_id], mx.array(final_ranking)), axis=0
        )
        if self.uniform_init:
            rank_prob = mx.concatenate(
                (mx.full((1, 4), 0.25), rank_prob[1:]), axis=0
            )
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(
            player_id, grp_feature, rank_by_player
        )
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        mx.eval(reward)
        return np.array(reward)

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        seq = np.concatenate(
            (
                grp_feature[:, 3 + player_id] * 1e4,
                [final_scores[player_id]],
            )
        )
        return seq[1:] - seq[:-1]
