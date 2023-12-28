import pickle
import time
import numpy as np
from tqdm import tqdm
import os
from numba import njit


def create_dir(create_dirs):
    """
    create necessary dirs.
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            # logger.info('Create dir: %s' % dir)
            print('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))


def get_novelty(item_novelty, all_item_array):
    print("getting novelty...")
    map_rawId_to_newId = dict(zip(item_novelty.index, np.arange(len(item_novelty))))
    time_start = time.time()
    item_novelty = np.array(item_novelty)
    all_item_array.astype(int)
    # res = np.array(list(map(lambda x: [item_novelty[i] for i in x], all_item_array)))
    func = lambda x: item_novelty[map_rawId_to_newId[x]]
    func = np.vectorize(func)
    res = func(all_item_array)
    res = res.sum(axis=1)
    time_end = time.time()
    print(f"got novelty in {time_end - time_start} seconds!")
    return res

def get_novelty2(item_novelty, all_item_array):
    print("getting novelty...")
    time_start = time.time()
    res = np.array(list(map(lambda x: [item_novelty[i] for i in x], all_item_array)))
    res = res.sum(axis=1)
    time_end = time.time()
    print(f"got novelty in {time_end - time_start} seconds!")
    return res

# def get_serendipity(target_df, df_item, to_go_item_array, to_go_rating_array,like_threshold, tag_label="tags"):
#     res = np.zeros(len(target_df))
#     for k, (seq, rating) in tqdm(
#         enumerate(zip(to_go_item_array, to_go_rating_array)),
#         total=len(to_go_item_array),
#         desc="computing serendipity",
#     ):
#         # for k, (seq, rating) in enumerate(zip(to_go_item_array, to_go_rating_array)):
#         cats = df_item.loc[seq][tag_label]
#         hist_cat = set()
#         brand_new = np.zeros(len(seq))
#         for ind, cat in enumerate(cats):
#             if any([each_cat not in hist_cat for each_cat in cat]):
#                 brand_new[ind] = True
#             else:
#                 brand_new[ind] = False
#             for each_cat in cat:
#                 hist_cat.add(each_cat)

#         is_like = rating >= like_threshold

#         # compute and operation of two numpy array
#         new_and_like = np.logical_and(brand_new, is_like)

#         res[k] = new_and_like.sum()
#     return res


@njit
def compute_serendipity(to_go_item_array, to_go_rating_array, indices, list_tags_items, like_threshold):
    
    map_rawId_to_newId = dict(zip(indices, np.arange(len(list_tags_items))))
    res = np.zeros(len(to_go_item_array))

    for k, (seq, rating) in enumerate(zip(to_go_item_array, to_go_rating_array)):
        # cats = list_tags_items.loc[seq][tag_label]

        hist_cat = []
        brand_new = np.zeros(len(seq))

        for idx, item in enumerate(seq):
            item_transformed = map_rawId_to_newId[item]
            cats = list(list_tags_items[item_transformed])

            for cat in cats:
                if cat not in hist_cat:
                    brand_new[idx] = True
                
            hist_cat.extend(cats)

        is_like = rating >= like_threshold
        # compute and operation of two numpy array
        new_and_like = np.logical_and(brand_new, is_like)

        res[k] = new_and_like.sum()
    return res


def get_serendipity(series_tags_items, to_go_item_array, to_go_rating_array, like_threshold):
    indices = series_tags_items.index.to_numpy()
    
    list_tags_items = series_tags_items.to_list()
    list_tags_items = [x.astype(int) for x in list_tags_items]
    
    print(f"computing serendipity for threshold {like_threshold} ...")
    res = compute_serendipity(to_go_item_array, to_go_rating_array, indices, list_tags_items, like_threshold)
    print(f"computed done for {like_threshold} ...")
    return res


def get_diversiy(series_tags_items, to_go_item_array):
    indices = series_tags_items.index.to_numpy()

    list_tags_items = series_tags_items.to_list()
    # list_tags_items = [list(x) for x in list_tags_items]
    list_tags_items = [x.astype(int) for x in list_tags_items]

    print("start computing diversity...")
    res = compute_diversity(to_go_item_array, indices, list_tags_items)
    print("diversity computed.")

    # map_rawId_to_newId = dict(zip(series_tags_items.index, np.arange(len(series_tags_items))))
    # total_num = len(to_go_item_array[0]) * (len(to_go_item_array[0]) - 1) / 2
    # res = np.zeros(len(to_go_item_array), dtype=np.float64)

    # div_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
    # for k, seq in tqdm(
    #     enumerate(to_go_item_array),
    #     total=len(to_go_item_array),
    #     desc="computing diversity",
    # ):
    #     # for k, seq in enumerate(all_item_ind_array):
    #     # cats = df_item.loc[seq]["genre_int"]
    #     # index = [item_index_dict[item] for item in seq]
    #     seq_transformed = [map_rawId_to_newId[item] for item in seq.tolist()]
    #     cats = series_tags_items[seq_transformed]
    #     # cats = np.zeros(len(seq))
    #     total_div = 0
    #     for ind, cat in enumerate(cats):
    #         # print(ind, item)
    #         # df_item.loc[item, "genre_int"] = cats
    #         for hist_cat in cats[:ind]:
    #             # print(hist_item)
    #             div = div_func(hist_cat, cat)
    #             total_div += div
    #     total_div /= total_num

    #     # print(total_div)
    #     res[k] = 1 - total_div
    return res


@njit
def compute_diversity(to_go_item_array, indices, list_tags_items):
    # list_tags_items = [list(x) for x in list_tags_items]

    map_rawId_to_newId = dict(zip(indices, np.arange(len(list_tags_items))))
    total_num = len(to_go_item_array[0]) * (len(to_go_item_array[0]) - 1) / 2
    res = np.zeros(len(to_go_item_array), dtype=np.float64)

    div_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))

    # for k, seq in tqdm(
    #     enumerate(to_go_item_array),
    #     total=len(to_go_item_array),
    #     desc="computing diversity",
    # ):
    for k, seq in enumerate(to_go_item_array):
        # for k, seq in enumerate(all_item_ind_array):
        # cats = df_item.loc[seq]["genre_int"]
        # index = [item_index_dict[item] for item in seq]
        seq_transformed = [map_rawId_to_newId[item] for item in seq]
        # cats = list_tags_items[seq_transformed]
        cats = [list_tags_items[item] for item in seq_transformed]

        # cats = np.zeros(len(seq))
        total_div = 0.0
        for ind, cat in enumerate(cats):
            # print(ind, item)
            # df_item.loc[item, "genre_int"] = cats
            for hist_cat in cats[:ind]:
                # print(hist_item)
                if len(hist_cat) == 0 or len(cat) == 0:
                    div = 0
                else:
                    div = div_func(list(hist_cat), list(cat))
                total_div += div
        total_div /= total_num

        # print(total_div)
        res[k] = 1 - total_div
        # print("done!")
    return res

def get_statistics(df, df_user, df_item, target_df, to_go_seq_dict, serendipity_threshold=[3, 4, 5], tag_label="tags"):
    # get the number of each item's consumption by users in df
    item_count = df.groupby("item_id").agg(len)["user_id"]
    # item_count_set = df.groupby("item_id").agg(lambda x: len(set(x)))["user_id"]

    num_users = len(df_user)

    # get novelty of each item
    item_novelty = item_count.apply(lambda x: np.log(num_users / x))

    # current_item_2d = np.expand_dims(seq_df["item_id"].to_numpy(), axis=1)
    # all_item_array = np.concatenate((seq_dict["item_id_list"], current_item_2d), axis=1)
    to_go_item_array = to_go_seq_dict["item_id_list"].astype(int)

    # current_rating_2d = np.expand_dims(seq_df["rating"].to_numpy(), axis=1)
    # all_rating_array = np.concatenate(
    #     (seq_dict["rating_list"], current_rating_2d), axis=1
    # )
    to_go_rating_array = to_go_seq_dict["rating_list"]

    # map a function to each element in a numpy ndarray and return a new ndarray
    target_df.loc[:, "novelty"] = get_novelty(item_novelty, to_go_item_array)

    # item_index_dict = dict(
    #     zip(df_item.index.to_numpy().astype(int), np.arange(len(df_item)))
    # )

    # func_each = lambda x: item_index_dict[x]
    # func_each = np.vectorize(func_each)
    # all_item_ind_array = func_each(to_go_item_array)

    series_tags_items = df_item[tag_label]

    target_df.loc[:, "diversity"] = get_diversiy(series_tags_items, to_go_item_array)

    # plot the distplot of seq_df["rating"]
    # sns.displot(target_df["rating"], kde=False, rug=False)
    # plt.savefig(os.path.join(FIGPATH, "rating_distplot.png"))
    # plt.close()

    # get the numpy format of cumulative distribution of seq_df["rating"], make sure the maximum is 1
    rating_cumsum = np.cumsum(
        target_df["rating"].value_counts().sort_index().to_numpy()
    ) / len(target_df)
    print(rating_cumsum)

    if type(serendipity_threshold) == list:
        for like_threshold in serendipity_threshold:
            target_df.loc[:, f"serendipity_{like_threshold}"] = get_serendipity(
                series_tags_items, to_go_item_array, to_go_rating_array, like_threshold
            )
    else:
        like_threshold = serendipity_threshold
        target_df.loc[:, f"serendipity_{like_threshold}"] = get_serendipity(
            series_tags_items, to_go_item_array, to_go_rating_array, like_threshold
        )

    target_df.loc[:, "sum_rating"] = to_go_rating_array.sum(axis=1)

    return target_df


def get_seq_target_and_indices(df, seq_columns, max_item_list_len, len_reward_to_go):
    last_uid = None
    uid_list, hist_item_list_indices, to_go_item_list_indices, target_indices, item_list_length = [], [], [], [], []
    seq_start = 0
    for i, uid in enumerate(tqdm(df["user_id"].to_numpy(), desc="getting sequences' indices...", total=len(df))):
        if last_uid != uid:
            last_uid = uid
            seq_start = i
        else:
            if i - seq_start > max_item_list_len:
                seq_start += 1

            target_idx = i - len_reward_to_go

            uid_list.append(uid)
            hist_item_list_indices.append((seq_start, target_idx))
            to_go_item_list_indices.append((target_idx, i))
            target_indices.append(target_idx)
            item_list_length.append(i - seq_start)

    uid_list = np.array(uid_list)
    hist_item_list_indices = np.array(hist_item_list_indices)
    to_go_item_list_indices = np.array(to_go_item_list_indices)
    target_indices = np.array(target_indices)
    item_list_length = np.array(item_list_length, dtype=np.int64)

    # Only use the part of sequences that satisfied the minimum length!
    reserved_idx = item_list_length >= len_reward_to_go

    uid_list = uid_list[reserved_idx]
    hist_item_list_indices = hist_item_list_indices[reserved_idx]
    to_go_item_list_indices = to_go_item_list_indices[reserved_idx]
    target_indices = target_indices[reserved_idx]
    item_list_length = item_list_length[reserved_idx]

    new_length = len(hist_item_list_indices)
    target_df = df.iloc[target_indices]
    target_df.reset_index(drop=True, inplace=True)

    def get_seq_field_data():
        hist_seq_dict = dict()
        to_go_seq_dict = dict()

        hist_shape = (new_length, max_item_list_len - len_reward_to_go)
        to_go_shape = (new_length, len_reward_to_go)

        cols = [col.name for col in seq_columns]
        for field in cols:
            if field != "user_id":
                list_field = f"{field}_list"

                hist_seq_dict[list_field] = np.zeros(hist_shape)
                to_go_seq_dict[list_field] = np.zeros(to_go_shape)

                values = df[field].to_numpy()
                print(f"getting sequences field data: {field} ...")

                res_hist_seq_field = hist_seq_dict[list_field]
                res_to_go_seq_field = to_go_seq_dict[list_field]

                get_seq_field(
                    res_hist_seq_field,
                    res_to_go_seq_field,
                    len_reward_to_go,
                    values,
                    hist_item_list_indices,
                    to_go_item_list_indices,
                    item_list_length,
                )

        return hist_seq_dict, to_go_seq_dict

    hist_seq_dict, to_go_seq_dict = get_seq_field_data()

    return target_df, hist_seq_dict, to_go_seq_dict


@njit
def get_seq_field(
        res_hist_seq_field,
        res_to_go_seq_field,
        len_reward_to_go,
        values,
        hist_item_list_indices,
        to_go_item_list_indices,
        item_list_length,
):
    for i, (hist_slice, to_go_slice, length) in enumerate(
            zip(hist_item_list_indices, to_go_item_list_indices, item_list_length)
    ):
        res_hist_seq_field[i][: length - len_reward_to_go] = values[hist_slice[0]:hist_slice[1]]
        res_to_go_seq_field[i] = values[to_go_slice[0]: to_go_slice[1]]

