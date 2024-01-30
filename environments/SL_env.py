import os
import pickle

import logzero
import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from numba import njit

from .utils import get_novelty, get_serendipity, get_diversiy
from .get_pareto_front import get_kde_density, get_pareto_front


class SLEnv:
    def __init__(self, item_columns, reward_columns):

        item_features = [col.name for col in item_columns]
        self.item_col = item_features.index("item_id")

        reward_features = [col.name for col in reward_columns]
        self.rating_col = reward_features.index("sum_rating")

        self.reward_columns = reward_columns
        self.target_features = [column.name for column in reward_columns]



    def reset(self, ):
        pass

    def step(self, buffer, eval_round):
        batch, indices = buffer.sample(0)
        seq_batch = batch.seq_batch
        seq_batch = seq_batch.reshape(buffer.buffer_num, -1, *seq_batch.shape[2:])

        # indices[:len_data_batch[0,0]] # todo: adjust the indices

        to_go_item_array = seq_batch[:, -1, self.item_col, -eval_round:].astype(int)  # TODO: Note: dim2 = 0 indicates the item_id
        to_go_rating_array = seq_batch[:, -1, self.rating_col,-eval_round:]  # TODO: note that the last index (-1) indicates the rating column!

        res = self.evaluate(to_go_item_array, to_go_rating_array)

        return res

    def compile_test(self, df_data, df_user, df_item, tag_label="tags"):
        self.df_data = df_data
        self.df_user = df_user
        self.df_item = df_item
        self.tag_label = tag_label

        # labelencoder for movielens data
        self.lbe_user = LabelEncoder()
        self.lbe_user.fit(df_user.index)
        self.lbe_item = LabelEncoder()
        self.lbe_item.fit(df_item.index)

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


