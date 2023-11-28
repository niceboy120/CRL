import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

from inputs import SparseFeatP
from ..utils import get_novelty, get_serendipity, get_diversiy
from ..base_env import BaseEnv


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")

class ML1MEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self.RESULTPATH = RESULTPATH
        self.FIGPATH = FIGPATH
        self.DATAPATH = DATAPATH
        super().__init__(*args, **kwargs)

    def step(self, buffer, eval_round):
        batch, indices = buffer.sample(0)
        (x_batch, seq_batch, y_batch, len_data_batch) = batch.x_batch, batch.seq_batch, batch.y_batch, batch.len_data_batch

        # x_batch = x_batch.reshape(buffer.buffer_num, -1, *x_batch.shape[2:])
        seq_batch = seq_batch.reshape(buffer.buffer_num, -1, *seq_batch.shape[2:])
        # y_batch = y_batch.reshape(buffer.buffer_num, -1, *y_batch.shape[2:])
        # len_data_batch = len_data_batch.reshape(buffer.buffer_num, -1, *len_data_batch.shape[2:])[:, -1].astype(int)

        # indices[:len_data_batch[0,0]] # todo: adjust the indices

        to_go_item_array = seq_batch[:, -1, 0, -eval_round:].astype(int)  # TODO: Note: dim2 = 0 indicates the item_id
        to_go_rating_array = seq_batch[:, -1, -1, -eval_round:]  # TODO: note that the last index (-1) indicates the rating column!

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





