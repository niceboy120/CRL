from functools import partial
from itertools import chain
import multiprocessing
import time
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
from tqdm import tqdm
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

from ..base import BaseEnv
from ..utils import get_diversiy, get_novelty, get_seq_dict, get_serendipity


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")

for path in [FIGPATH, RESULTPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class KuaiRand1KEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self.DATAPATH = DATAPATH
        self.FIGPATH = FIGPATH
        self.RESULTPATH = RESULTPATH
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_features(use_userinfo=False):
        user_features = ["user_id", "user_active_degree", "is_live_streamer", "is_video_author", "follow_user_num_range",
                         "fans_user_num_range", "friend_user_num_range", "register_days_range",] + [f"onehot_feat{x}" for x in range(18)]
        if not use_userinfo:
            user_features = ["user_id"]
        item_features = (["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"])
        reward_features = ["is_click"]
        return user_features, item_features, reward_features
    
    @staticmethod
    def load_user_feat():
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features_1k.csv')
        df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                                 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                                 'fans_user_num_range', 'friend_user_num_range',
                                                 'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                              )
        for col in ['user_active_degree',
                    'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                    'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

            df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1
        for col in [f'onehot_feat{x}' for x in range(18)]:
            df_user.loc[df_user[col].isna(), col] = -124
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1

        df_user = df_user.set_index("user_id")
        return df_user
    
    @staticmethod
    def load_category():
        file_path_df_feat = os.path.join(DATAPATH, 'df_item_1k_processed.csv')
        file_path_list_feat = os.path.join(DATAPATH, 'list_feat_1k_processed.pkl')
        if os.path.exists(file_path_df_feat) and os.path.exists(file_path_list_feat):
            df_feat = pd.read_csv(file_path_df_feat, index_col="item_id")
            list_feat_num = pickle.load(open(file_path_list_feat, "rb"))
            
        else:
            print("load item feature")
            filepath = os.path.join(DATAPATH, 'video_features_basic_1k.csv')
            df_item = pd.read_csv(filepath, usecols=["video_id", "tag"], dtype=str, index_col="video_id")
            ind = df_item['tag'].isna()
            df_item['tag'].loc[~ind] = df_item['tag'].loc[~ind].map(lambda x: eval(f"[{x}]"))
            df_item['tag'].loc[ind] = [[-1]] * ind.sum()

            df_item_all = df_item.reindex((list(range(df_item.index.min(),df_item.index.max()+1))), fill_value=[-1], copy=False)
            # b = set(range(df_item.index.max())) - set(df_item.index)
            # b = np.array(list(b))
            # df_item_all.loc[b]

            list_feat = df_item_all['tag'].to_list()
            list_feat_num = list(map(lambda ll: [x + 1 for x in ll],list_feat))

            df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3', 'feat4'])
            df_feat.index.name = "item_id"
            df_feat[df_feat.isna()] = -1
            df_feat = df_feat + 1
            df_feat = df_feat.astype(int)

            df_feat.to_csv(file_path_df_feat)
            pickle.dump(list_feat_num, open(file_path_list_feat, "wb"))

        return list_feat_num, df_feat

    @staticmethod
    def load_item_feat():
        
        list_feat_num, df_feat = KuaiRand1KEnv.load_category()
        # video_mean_duration = KuaiRand1KEnv.load_video_duration()
        # df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")
        df_item = df_feat

        return df_item

    @staticmethod
    def get_seq_data(max_item_list_len, len_reward_to_go, reload):
        seq_columns=[
            "user_id",
            "item_id",
            # "time_ms",
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "play_time_ms",
            # "duration_ms",
            # "profile_stay_time",
            # "comment_stay_time",
            # "is_profile_enter",
            # "is_rand",
        ]
        df_seq_rewards, hist_seq_dict, to_go_seq_dict = BaseEnv.get_and_save_seq_data(KuaiRand1KEnv, RESULTPATH, seq_columns, max_item_list_len, len_reward_to_go, reload=reload)
        return df_seq_rewards, hist_seq_dict, to_go_seq_dict


    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()

        self.action = None  # Add by Chongming
        self._reset_history()

        return self.state
    
    def step(self, action):
        self.action = action
        reward = None,
        done = None,
        return self.state, reward, done, {'cum_reward': self.cum_reward}



def get_df_kuairand_1k(inter_path):
    df = pd.read_csv(
        inter_path,
        # sep="\t",
        header=0,
        # dtype={0: int, 1: int, 2: float, 3: int},
        # names=["user_id", "item_id", "rating", "timestamp"],
        # nrows=100000,
    )
    print(df)

    df.sort_values(by=["user_id", "time_ms"], ascending=True, inplace=True)

    # transfer timestamp to datetime
    df["date"] = pd.to_datetime(df["time_ms"], unit="ms")

    # transfer datetime to date
    df["day"] = df["date"].dt.date

    df.groupby(["user_id", "day"]).agg(len).describe()
    df.groupby(["user_id"]).agg(len).describe()

    return df


def get_user_item_feat(user_feat_path, item_feat_path):
    df_user = pd.read_csv(
        user_feat_path,
        # sep="\t",
        header=0,
        # dtype={0: int, 1: int, 2: str, 3: int, 4: str},
    )
    # remove the last 6 tokens in each column name of df_user
    # df_user.columns = df_user.columns.str[:-6]

    df_item = pd.read_csv(
        item_feat_path,
        # sep="\t",
        header=0,
        # names=["item_id", "movie_title", "release_year", "genre"],
        # dtype={0: int, 1: str, 2: int, 3: str},
    )

    # split df_item["genre"] into a list of genres
    # df_item["genre"] = df_item["genre"].apply(lambda x: x.split())
    # df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))
    df_item["tags"] = df_item["tag"].apply(
        lambda x: [-1] if type(x) is float else [int(y) for y in x.split(",")]
    )
    df_item["num_tags"] = df_item["tags"].apply(lambda x: len(x))

    return df_user, df_item





def get_statistics(df, df_user, df_item, seq_df, seq_dict, tag_label = "tags"):
    # get the number of each item's consumption by users in df
    item_count = df.groupby("item_id").agg(len)["user_id"]
    # item_count_set = df.groupby("item_id").agg(lambda x: len(set(x)))["user_id"]

    num_users = len(df_user)

    # get novelty of each item
    item_novelty = item_count.apply(lambda x: np.log(num_users / x))

    current_item_2d = np.expand_dims(seq_df["item_id"].to_numpy(), axis=1)
    all_item_array = np.concatenate((seq_dict["item_id_list"], current_item_2d), axis=1)

    metric_list = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",]
    
    all_metric_dict = {}
    for metric in metric_list:
        current_metric_2d = np.expand_dims(seq_df[metric].to_numpy(), axis=1)
        all_metric_array = np.concatenate((seq_dict[metric + "_list"], current_metric_2d), axis=1)
        all_metric_dict[metric] = all_metric_array


    # current_rating_2d = np.expand_dims(seq_df["rating"].to_numpy(), axis=1)
    # all_rating_array = np.concatenate(
    #     (seq_dict["is_like_list"], current_rating_2d), axis=1
    # )

    df_item.set_index("item_id", inplace=True)

    # map a function to each element in a numpy ndarray and return a new ndarray
    seq_df["novelty"] = get_novelty(item_novelty, all_item_array)

    # cast all_item_array to int type
    all_item_array = all_item_array.astype(int)
    item_index_dict = dict(zip(df_item.index.to_numpy().astype(int), np.arange(len(df_item))))

    func_each = lambda x: item_index_dict[x]
    func_each = np.vectorize(func_each)
    all_item_ind_array = func_each(all_item_array)
    # for seq in all_item_array:
    #     for item in seq:
    #         all_item_ind_array item_index_dict

    df_item_tags = df_item[tag_label].to_numpy()
    # compute the diversity of each sequence:
    # seq_df["diversity2"] = get_diversiy_multiprocessing(df_item_genre, all_item_ind_array)
    seq_df["diversity"] = get_diversiy(df_item_tags, all_item_ind_array)


    # get the numpy format of cumulative distribution of seq_df["rating"], make sure the maximum is 1
    click_cumsum = np.cumsum(seq_df["is_click"].value_counts().sort_index().to_numpy()) / len(seq_df)
    print("click cumulative distribution:", click_cumsum)

    like_cumsum = np.cumsum(seq_df["is_like"].value_counts().sort_index().to_numpy()) / len(seq_df)
    print("like cumulative distribution:", like_cumsum)

    longview_cumsum = np.cumsum(seq_df["long_view"].value_counts().sort_index().to_numpy()) / len(seq_df)
    print("longview cumulative distribution:", longview_cumsum)

    seq_df["serendipity_click"] = get_serendipity(seq_df, df_item, all_item_array, all_metric_dict["is_click"], 1, tag_label)

    seq_df["serendipity_like"] = get_serendipity(seq_df, df_item, all_item_array, all_metric_dict["is_like"], 1, tag_label)

    seq_df["serendipity_longview"] = get_serendipity(seq_df, df_item, all_item_array, all_metric_dict["long_view"], 1, tag_label)

    for metric in all_metric_dict.keys():
        seq_df[f"sum_{metric}"] = all_metric_dict[metric].sum(axis=1)

    # seq_df["sum_rating"] = all_rating_array.sum(axis=1)

    return seq_df




def get_pairgrid_plot(seq_df, filepath_pairgrid):

    df_visual = seq_df[["novelty", "diversity", "serendipity_longview", 'sum_is_click', 'sum_is_like', 'sum_is_follow',
       'sum_is_comment', 'sum_is_forward', 'sum_long_view']].iloc[:10000]

    g = sns.PairGrid(df_visual, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    plt.savefig(filepath_pairgrid, bbox_inches="tight", pad_inches=0)
    plt.close()



def get_seq_df_rts(columns, max_item_list_len = 10):
    inter_path = os.path.join(DATAPATH, "log_standard_4_22_to_5_08_1k.csv")
    user_feat_path = os.path.join(DATAPATH, "user_features_1k.csv")
    item_feat_path = os.path.join(DATAPATH, "video_features_basic_1k.csv")

    df = get_df_kuairand_1k(inter_path)
    df_user, df_item = get_user_item_feat(user_feat_path, item_feat_path)

    df.rename(columns={"video_id": "item_id"}, inplace=True)
    df_item.rename(columns={"video_id": "item_id"}, inplace=True)

    
    seq_dict_filepath = os.path.join(DATAPATH, "seq_dict.pkl")
    seq_df, seq_dict = get_seq_dict(df, columns, max_item_list_len, seq_dict_filepath)

    filepath_seq_df_rts = os.path.join(DATAPATH, "seq_df_rts.csv")
    if os.path.exists(filepath_seq_df_rts):
        seq_df_rts = pd.read_csv(filepath_seq_df_rts)
    else:
        seq_df_rts = get_statistics(df, df_user, df_item, seq_df, seq_dict)
        seq_df_rts.to_csv(filepath_seq_df_rts, index=False)

    return seq_df_rts

    # filepath_pairgrid = os.path.join(DATAPATH, "pairgrid_plot.pdf")
    # get_pairgrid_plot(seq_df, filepath_pairgrid)


if __name__ == "__main__":
    columns = ["user_id", "item_id", "time_ms", "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
               "long_view", "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter", "is_rand",]
    seq_df_rts = get_seq_df_rts(columns)
    