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



