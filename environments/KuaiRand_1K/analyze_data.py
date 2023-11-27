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


max_item_list_len = 10


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


def get_seq_dict(seq_dict_filepath):
    last_uid = None
    uid_list, item_list_index, target_index, item_list_length = [], [], [], []
    seq_start = 0
    for i, uid in enumerate(df["user_id"].to_numpy()):
        if last_uid != uid:
            last_uid = uid
            seq_start = i
        else:
            if i - seq_start > max_item_list_len:
                seq_start += 1
            uid_list.append(uid)
            item_list_index.append(slice(seq_start, i))
            target_index.append(i)
            item_list_length.append(i - seq_start)

    uid_list = np.array(uid_list)
    item_list_index = np.array(item_list_index)
    target_index = np.array(target_index)
    item_list_length = np.array(item_list_length, dtype=np.int64)

    full_max_list = item_list_length == max_item_list_len

    uid_list = uid_list[full_max_list]
    item_list_index = item_list_index[full_max_list]
    target_index = target_index[full_max_list]
    item_list_length = item_list_length[full_max_list]

    new_length = len(item_list_index)

    df.reset_index(drop=True, inplace=True)
    seq_df = df.iloc[target_index]

    def get_from_scratch(seq_dict_filepath):
        seq_dict = dict()
        cols = [
            "user_id",
            "item_id",
            "time_ms",
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "play_time_ms",
            "duration_ms",
            "profile_stay_time",
            "comment_stay_time",
            "is_profile_enter",
            "is_rand",
        ]

        for field in cols:
            if field != "user_id":
                list_field = f"{field}_list"
                list_len = max_item_list_len
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                seq_dict[list_field] = np.zeros(shape)

                value = df[field]
                for i, (index, length) in tqdm(
                    enumerate(zip(item_list_index, item_list_length)),
                    total=len(item_list_index), desc=f"{field}",
                ):
                    seq_dict[list_field][i][:length] = value[index]

        # pickle save the seq_dict
        pickle.dump(seq_dict, open(seq_dict_filepath, "wb"))
        return seq_dict

    if os.path.exists(seq_dict_filepath):
        seq_dict = pickle.load(open(seq_dict_filepath, "rb"))
    else:
        seq_dict = get_from_scratch(seq_dict_filepath)

    return seq_df, seq_dict


# def get_novelty(item_novelty, all_item_array):
#     print("getting novelty...")
#     time_start = time.time()
#     res = np.array(list(map(lambda x: [item_novelty[i] for i in x], all_item_array)))
#     res = res.sum(axis=1)
#     time_end = time.time()
#     print(f"got novelty in {time_end - time_start} seconds!")
#     return res


def get_novelty(item_novelty, all_item_array):
    print("getting novelty...")
    time_start = time.time()
    # res = np.array(list(map(lambda x: [item_novelty[i] for i in x], all_item_array)))
    func = lambda x: item_novelty[x]
    func = np.vectorize(func)
    res = func(all_item_array)
    res = res.sum(axis=1)
    time_end = time.time()
    print(f"got novelty in {time_end - time_start} seconds!")
    return res


def get_diversiy(df_item_tags, all_item_ind_array):
    div_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))

    total_num = len(all_item_ind_array[0]) * (len(all_item_ind_array[0]) - 1) / 2
    res = np.zeros(len(all_item_ind_array), dtype=np.float64)

    for k, seq in tqdm(
        enumerate(all_item_ind_array),
        total=len(all_item_ind_array),
        desc="computing diversity",
    ):
        # for k, seq in enumerate(all_item_ind_array):
        # cats = df_item.loc[seq]["genre_int"]
        # index = [item_index_dict[item] for item in seq]
        cats = df_item_tags[seq]
        # cats = np.zeros(len(seq))
        total_div = 0
        for ind, cat in enumerate(cats):
            # print(ind, item)
            # df_item.loc[item, "genre_int"] = cats
            for hist_cat in cats[:ind]:
                # print(hist_item)
                div = div_func(hist_cat, cat)
                total_div += div
        total_div /= total_num

        # print(total_div)
        res[k] = 1 - total_div
    return res


def processing_func(seq, df_item_genre):
    div_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
    cats = df_item_genre[seq]
    total_num = len(seq) * (len(seq) - 1) / 2
    total_div = 0
    for ind, cat in enumerate(cats):
        for hist_cat in cats[:ind]:
            div = div_func(hist_cat, cat)
            total_div += div
    total_div /= total_num
    return 1 - total_div


def get_diversiy_multiprocessing(df_item_genre, all_item_ind_array):
    div_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))

    total_num = len(all_item_ind_array[0]) * (len(all_item_ind_array[0]) - 1) / 2
    res = np.zeros(len(all_item_ind_array), dtype=np.float64)

    # use multiprocessing to speed up
    num_processes = multiprocessing.cpu_count()
    pool = Pool(num_processes // 2)
    # res = pool.map(partial(processing_func, df_item_genre=processing_func), all_item_ind_array)

    res_list = []
    progress_bar = tqdm(total=len(all_item_ind_array), desc="computing diversity")
    for result in pool.imap(
        partial(processing_func, df_item_genre=df_item_genre), all_item_ind_array
    ):
        progress_bar.update(1)
        res_list.append(result)
    res = res_list.to_numpy()
    # for k, seq in tqdm(
    #     enumerate(all_item_ind_array), total=len(all_item_ind_array), desc="computing diversity"
    # ):
    #     cats = df_item_genre[seq]
    #     total_div = 0
    #     for ind, cat in enumerate(cats):
    #         for hist_cat in cats[:ind]:
    #             div = div_func(hist_cat, cat)
    #             total_div += div
    #     total_div /= total_num

    #     # print(total_div)
    #     res[k] = 1 - total_div
    return res


def get_serendipity(seq_df, df_item, all_item_array, all_rating_array, like_threshold):
    res = np.zeros(len(seq_df))
    for k, (seq, rating) in tqdm(
        enumerate(zip(all_item_array, all_rating_array)),
        total=len(all_item_array),
        desc="computing serendipity",
    ):
        # for k, (seq, rating) in enumerate(zip(all_item_array, all_rating_array)):
        cats = df_item.loc[seq]["tags"]
        hist_cat = set()
        brand_new = np.zeros(len(seq))
        for ind, cat in enumerate(cats):
            if any([each_cat not in hist_cat for each_cat in cat]):
                brand_new[ind] = True
            else:
                brand_new[ind] = False
            for each_cat in cat:
                hist_cat.add(each_cat)

        is_like = rating >= like_threshold

        # compute and operation of two numpy array
        new_and_like = np.logical_and(brand_new, is_like)

        res[k] = new_and_like.sum()
    return res


def get_statistics(df, df_user, df_item, seq_df, seq_dict, filepath_seq_df):
    # get the number of each item's consumption by users in df
    item_count = df.groupby("item_id").agg(len)["user_id"]
    # item_count_set = df.groupby("item_id").agg(lambda x: len(set(x)))["user_id"]

    num_users = len(df_user)

    # get novelty of each item
    item_novelty = item_count.apply(lambda x: np.log(num_users / x))

    current_item_2d = np.expand_dims(seq_df["item_id"].to_numpy(), axis=1)
    all_item_array = np.concatenate((seq_dict["item_id_list"], current_item_2d), axis=1)

    metric_list = ["is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "long_view",]
    
    
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
    item_index_dict = dict(
        zip(df_item.index.to_numpy().astype(int), np.arange(len(df_item)))
    )

    func_each = lambda x: item_index_dict[x]
    func_each = np.vectorize(func_each)
    all_item_ind_array = func_each(all_item_array)
    # for seq in all_item_array:
    #     for item in seq:
    #         all_item_ind_array item_index_dict

    df_item_tags = df_item["tags"].to_numpy()
    # compute the diversity of each sequence:
    # seq_df["diversity2"] = get_diversiy_multiprocessing(df_item_genre, all_item_ind_array)
    seq_df["diversity"] = get_diversiy(df_item_tags, all_item_ind_array)

    # # plot the distplot of seq_df["rating"]
    # sns.displot(seq_df["rating"], kde=False, rug=False)
    # plt.savefig("dataset/ml-1m/rating_distplot.png")
    # plt.close()

    # get the numpy format of cumulative distribution of seq_df["rating"], make sure the maximum is 1
    click_cumsum = np.cumsum(
        seq_df["is_click"].value_counts().sort_index().to_numpy()
    ) / len(seq_df)
    print("click cumulative distribution:", click_cumsum)

    like_cumsum = np.cumsum(
        seq_df["is_like"].value_counts().sort_index().to_numpy()
    ) / len(seq_df)
    print("like cumulative distribution:", like_cumsum)

    longview_cumsum = np.cumsum(
        seq_df["long_view"].value_counts().sort_index().to_numpy()
    ) / len(seq_df)
    print("longview cumulative distribution:", longview_cumsum)

    seq_df["serendipity_click"] = get_serendipity(
        seq_df, df_item, all_item_array, all_metric_dict["is_click"], 1
    )

    seq_df["serendipity_like"] = get_serendipity(
        seq_df, df_item, all_item_array, all_metric_dict["is_like"], 1
    )

    seq_df["serendipity_longview"] = get_serendipity(
        seq_df, df_item, all_item_array, all_metric_dict["long_view"], 1
    )

    for metric in all_metric_dict.keys():
        seq_df[f"sum_{metric}"] = all_metric_dict[metric].sum(axis=1)

    # seq_df["sum_rating"] = all_rating_array.sum(axis=1)

    # save seq_df
    seq_df.to_csv(filepath_seq_df, index=False)

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


if __name__ == "__main__":
    # get ml-1m df:
    inter_path = "dataset/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv"
    user_feat_path = "dataset/KuaiRand-1K/data/user_features_1k.csv"
    item_feat_path = "dataset/KuaiRand-1K/data/video_features_basic_1k.csv"

    df = get_df_kuairand_1k(inter_path)
    df_user, df_item = get_user_item_feat(user_feat_path, item_feat_path)

    df.rename(columns={"video_id": "item_id"}, inplace=True)
    df_item.rename(columns={"video_id": "item_id"}, inplace=True)

    seq_dict_filepath = "dataset/KuaiRand-1K/data/seq_dict.pkl"
    seq_df, seq_dict = get_seq_dict(seq_dict_filepath)

    filepath_seq_df = "dataset/KuaiRand-1K/data/seq_df_rts.csv"
    if os.path.exists(filepath_seq_df):
        seq_df = pd.read_csv(filepath_seq_df)
    else:
        seq_df = get_statistics(df, df_user, df_item, seq_df, seq_dict, filepath_seq_df=filepath_seq_df)

    filepath_pairgrid = "dataset/KuaiRand-1K/data/pairgrid_plot.pdf"
    # get_pairgrid_plot(seq_df, filepath_pairgrid)
