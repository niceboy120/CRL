import os
import pickle

import logzero
import numpy as np

import pandas as pd
from tqdm import tqdm
from numba import njit

from .utils import get_novelty, get_serendipity, get_diversiy
from .get_pareto_front import get_kde_density, get_pareto_front


class BaseEnv:
    def __init__(self, df_seq_rewards, target_features, seq_columns, bin=100, percentile=90, pareto_reload=False):

        features = [col.name for col in seq_columns]
        self.item_col = features.index("item_id")
        self.rating_col = features.index("rating")
        self.item_padding_id = None
        self.user_padding_id = None


        self.seq_columns = seq_columns
        self.target_features = target_features


        # 1. Directly indicate Pareto Fronts manually
        self.pareto_front = np.array([[0.9, 0.5], [0.5, 0.9], [0.8, 0.8]])


        # # 2. Compute Pareto Front via KDE on training data
        # pareto_front_path = f"pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        # raw_pareto_front_path = f"raw_pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        # pareto_front_filepath = os.path.join(self.RESULTPATH, pareto_front_path)
        # raw_pareto_front_filepath = os.path.join(self.RESULTPATH, raw_pareto_front_path)
        # if not pareto_reload and os.path.exists(raw_pareto_front_filepath): # use raw pareto front
        #     print("raw pareto front has already been computed! Loading...")
        #     with open(raw_pareto_front_filepath, "rb") as f:
        #         self.pareto_front = pickle.load(f)
        #     print("raw pareto front loaded!")
        #     # return self.raw_pareto_front
        # else:
        #     density_flat, grid_flat = get_kde_density(df_seq_rewards, target_features, bin)
        #     self.pareto_front = get_pareto_front(density_flat, grid_flat, percentile)
        #     print(f"pareto front for bin:{bin} and percentile:{percentile} is ", self.pareto_front)
        #     # pickle dump pareto_front
        #     with open(pareto_front_filepath, "wb") as f:
        #         pickle.dump(self.pareto_front, f)


        logzero.logger.info(f"All Pareto Fronts for Env are: {self.pareto_front}")

    def reset(self, ):
        pass

    def step(self, buffer, eval_round):
        batch, indices = buffer.sample(0)
        seq_batch = batch.seq_batch
        seq_batch = seq_batch.reshape(buffer.buffer_num, -1, *seq_batch.shape[2:])

        # indices[:len_data_batch[0,0]] # todo: adjust the indices

        to_go_item_array = seq_batch[:, -1, self.item_col, -eval_round:].astype(
            int)  # TODO: Note: dim2 = 0 indicates the item_id
        to_go_rating_array = seq_batch[:, -1, self.rating_col,
                             -eval_round:]  # TODO: note that the last index (-1) indicates the rating column!

        res = self.evaluate(to_go_item_array, to_go_rating_array)

        return res

    def compile_test(self, df_data, df_user, df_item, tag_label="tags"):
        self.df_data = df_data
        self.df_user = df_user
        self.df_item = df_item
        self.tag_label = tag_label

        item_count = self.df_data.groupby("item_id").agg(len)["user_id"]
        # item_count_set = df.groupby("item_id").agg(lambda x: len(set(x)))["user_id"]
        num_users = len(self.df_user)
        # get novelty of each item
        item_novelty = item_count.apply(lambda x: np.log(num_users / x))
        self.item_novelty = item_novelty.reindex(list(range(self.df_item.index.min(), self.df_item.index.max() + 1)))

        self.item_novelty.loc[self.item_novelty.isna()] = item_novelty.max()
        self.series_tags_items = self.df_item[self.tag_label]

    def evaluate(self, to_go_item_array, to_go_rating_array, ):

        novelty = get_novelty(self.item_novelty, to_go_item_array)
        # if "diversity" in self.target_features:
        diversity = get_diversiy(self.series_tags_items, to_go_item_array)
        # if "novelty" in self.target_features:
        # if "rating" in self.target_features:
        rating = to_go_rating_array.sum(axis=1)
        serendipity = get_serendipity(self.series_tags_items, to_go_item_array, to_go_rating_array, 4)

        res = {
            "novelty": novelty.mean(),
            "novelty_std": novelty.std(),
            "diversity": diversity.mean(),
            "diversity_std": diversity.std(),
            "rating": rating.mean(),
            "rating_std": rating.std(),
            "serendipity": serendipity.mean(),
            "serendipity_std": serendipity.std(),
        }
        return res


