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
dataset_dir_path = "environments/ML_1M"

def get_df_ml_1m(inter_path):
    df = pd.read_csv(
        inter_path,
        sep="\t",
        header=0,
        dtype={0: int, 1: int, 2: float, 3: int},
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    print(df)

    df.sort_values(by=["user_id", "timestamp"], ascending=True, inplace=True)

    # transfer timestamp to datetime
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")

    # transfer datetime to date
    df["day"] = df["date"].dt.date

    df.groupby(["user_id", "day"]).agg(len).describe()
    df.groupby(["user_id"]).agg(len).describe()

    return df


def get_user_item_feat(user_feat_path, item_feat_path):
    df_user = pd.read_csv(
        user_feat_path,
        sep="\t",
        header=0,
        dtype={0: int, 1: int, 2: str, 3: int, 4: str},
    )
    # remove the last 6 tokens in each column name of df_user
    df_user.columns = df_user.columns.str[:-6]

    df_item = pd.read_csv(
        item_feat_path,
        sep="\t",
        header=0,
        names=["item_id", "movie_title", "release_year", "genre"],
        dtype={0: int, 1: str, 2: int, 3: str},
    )

    # split df_item["genre"] into a list of genres
    df_item["genre"] = df_item["genre"].apply(lambda x: x.split())
    df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))

    return df_user, df_item


def get_seq_dict():
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
    new_df = df.iloc[target_index]

    def get_from_scratch():
        seq_dict = dict()
        cols = df.columns[:4]
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
                    total=len(item_list_index),
                ):
                    seq_dict[list_field][i][:length] = value[index]

        # pickle save the seq_dict
        pickle.dump(seq_dict, open(os.path.join(dataset_dir_path, "seq_dict.pkl"), "wb"))
        return seq_dict

    if os.path.exists(os.path.join(dataset_dir_path, "seq_dict.pkl")):
        seq_dict = pickle.load(open(os.path.join(dataset_dir_path, "seq_dict.pkl"), "rb"))
    else:
        seq_dict = get_from_scratch()

    return new_df, seq_dict


def get_novelty(item_novelty, all_item_array):
    print("getting novelty...")
    time_start = time.time()
    res = np.array(list(map(lambda x: [item_novelty[i] for i in x], all_item_array)))
    res = res.sum(axis=1)
    time_end = time.time()
    print(f"got novelty in {time_end - time_start} seconds!")
    return res


def get_novelty2(item_novelty, all_item_array):
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


def get_diversiy(df_item_genre, all_item_ind_array):
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
        cats = df_item_genre[seq]
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


def get_serendipity(new_df, df_item, all_item_array, all_rating_array, like_threshold):
    res = np.zeros(len(new_df))
    for k, (seq, rating) in tqdm(
        enumerate(zip(all_item_array, all_rating_array)),
        total=len(all_item_array),
        desc="computing serendipity",
    ):
        # for k, (seq, rating) in enumerate(zip(all_item_array, all_rating_array)):
        cats = df_item.loc[seq]["genre_int"]
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


def get_statistics(df, df_user, df_item, new_df, seq_dict):
    # get the number of each item's consumption by users in df
    item_count = df.groupby("item_id").agg(len)["user_id"]
    # item_count_set = df.groupby("item_id").agg(lambda x: len(set(x)))["user_id"]

    num_users = len(df_user)

    # get novelty of each item
    item_novelty = item_count.apply(lambda x: np.log(num_users / x))

    current_item_2d = np.expand_dims(new_df["item_id"].to_numpy(), axis=1)
    all_item_array = np.concatenate((seq_dict["item_id_list"], current_item_2d), axis=1)

    current_rating_2d = np.expand_dims(new_df["rating"].to_numpy(), axis=1)
    all_rating_array = np.concatenate(
        (seq_dict["rating_list"], current_rating_2d), axis=1
    )

    df_item.set_index("item_id", inplace=True)

    # get the transformed genre of each item
    lbe = LabelEncoder()
    a_list_lists = df_item["genre"].to_list()
    # chain a list of lists in a list
    a_list = list(chain(*a_list_lists))
    lbe.fit(a_list)
    df_item["genre_int"] = df_item["genre"].apply(lambda x: lbe.transform(x))

    # map a function to each element in a numpy ndarray and return a new ndarray
    new_df["novelty"] = get_novelty2(item_novelty, all_item_array)

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

    df_item_genre = df_item["genre_int"].to_numpy()
    # compute the diversity of each sequence:
    # new_df["diversity2"] = get_diversiy_multiprocessing(df_item_genre, all_item_ind_array)
    new_df["diversity"] = get_diversiy(df_item_genre, all_item_ind_array)

    # plot the distplot of new_df["rating"]
    sns.displot(new_df["rating"], kde=False, rug=False)
    plt.savefig(os.path.join(dataset_dir_path, "rating_distplot.png"))
    plt.close()

    # get the numpy format of cumulative distribution of new_df["rating"], make sure the maximum is 1
    rating_cumsum = np.cumsum(
        new_df["rating"].value_counts().sort_index().to_numpy()
    ) / len(new_df)
    print(rating_cumsum)

    like_threshold = 5
    new_df["serendipity_5"] = get_serendipity(
        new_df, df_item, all_item_array, all_rating_array, like_threshold
    )

    like_threshold = 4
    new_df["serendipity_4"] = get_serendipity(
        new_df, df_item, all_item_array, all_rating_array, like_threshold
    )

    like_threshold = 3
    new_df["serendipity_3"] = get_serendipity(
        new_df, df_item, all_item_array, all_rating_array, like_threshold
    )

    new_df["sum_rating"] = all_rating_array.sum(axis=1)

    # save new_df
    new_df.to_csv(os.path.join(dataset_dir_path, "new_df_done.csv"), index=False)

    return new_df


def visualize_3d(new_df):
    df_visual = new_df[["novelty", "diversity", "serendipity"]]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the values
    ax.scatter(
        df_visual["novelty"],
        df_visual["diversity"],
        df_visual["serendipity"],
        s=10,
        c="b",
        marker="s",
    )

    # Set labels and title
    ax.set_xlabel("novelty")
    ax.set_ylabel("diversity")
    ax.set_zlabel("serendipity")
    ax.set_title("3D Scatter Plot")

    plt.savefig(os.path.join(dataset_dir_path, "3d_scatter.pdf"), bbox_inches="tight", pad_inches=0)

    plt.show()


def visualize_2d(new_df):
    df_visual = new_df[["novelty", "diversity", "serendipity", "sum_rating"]].iloc[:10000]

    # Create a 2d scattor plot
    fig = plt.figure()

    ax1 = fig.add_subplot(131)

    # Plot the values
    # ax1.scatter(df_visual["novelty"], df_visual["diversity"], s=10, c="b", marker="s")
    sns.kdeplot(
        data=df_visual,
        x="novelty",
        y="diversity",
        levels=[0.8, 0.9],  # , thresh=.3,
        ax=ax1,
    )
    # Set labels and title
    ax1.set_xlabel("novelty")
    ax1.set_ylabel("diversity")

    ax2 = fig.add_subplot(132)
    # Plot the values
    # ax2.scatter(df_visual["novelty"], df_visual["serendipity"], s=10, c="b", marker="s")
    sns.kdeplot(
        data=df_visual,
        x="novelty",
        y="serendipity",
        levels=[0.8, 0.9],  # , thresh=.3,
        ax=ax2,
    )
    # Set labels and title
    ax2.set_xlabel("novelty")
    ax2.set_ylabel("serendipity")

    ax3 = fig.add_subplot(133)
    # Plot the values
    # ax3.scatter(df_visual["diversity"], df_visual["serendipity"], s=10, c="b", marker="s")
    sns.kdeplot(
        data=df_visual,
        x="diversity",
        y="serendipity",
        levels=[0.8, 0.9],  # , thresh=.3,
        ax=ax3,
    ) 
    # Set labels and title
    ax3.set_xlabel("diversity")
    ax3.set_ylabel("serendipity")

    plt.savefig(os.path.join(dataset_dir_path, "2d_scatters.pdf"), bbox_inches="tight", pad_inches=0)

    plt.close()



def get_pairgrid_plot(new_df):
    df_visual = new_df[["novelty", "diversity", "serendipity", "sum_rating"]].iloc[:10000]

    g = sns.PairGrid(df_visual, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    plt.savefig(os.path.join(dataset_dir_path, "pairgrid_plot.pdf"), bbox_inches="tight", pad_inches=0)
    plt.close()



if __name__ == "__main__":
    # get ml-1m df:

    inter_path = os.path.join(dataset_dir_path, "ml-1m.inter")
    user_feat_path = os.path.join(dataset_dir_path, "ml-1m.user")
    item_feat_path = os.path.join(dataset_dir_path, "ml-1m.item")

    df = get_df_ml_1m(inter_path)
    df_user, df_item = get_user_item_feat(user_feat_path, item_feat_path)

    new_df, seq_dict = get_seq_dict()

    if os.path.exists(os.path.join(dataset_dir_path, "new_df_done.csv")):
        new_df = pd.read_csv(os.path.join(dataset_dir_path, "new_df_done.csv"))
    else:
        new_df = get_statistics(df, df_user, df_item, new_df, seq_dict)

    new_df["serendipity"] = new_df["serendipity_5"]

    get_pairgrid_plot(new_df)
