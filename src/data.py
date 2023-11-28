import argparse
import os

import numpy as np
from inputs import DenseFeat, SparseFeat, SparseFeatP, VarLenSparseFeat
from state_action_return_dataset import StateActionReturnDataset


SRCPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(SRCPATH)


def get_common_args(args):
    env = args.env

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_userinfo", dest="use_userinfo", action="store_true")
    parser.add_argument(
        "--no_userinfo", dest="use_userinfo", action="store_false")

    parser.add_argument("--is_binarize", dest="is_binarize",
                        action="store_true")
    parser.add_argument("--no_binarize", dest="is_binarize",
                        action="store_false")

    parser.add_argument(
        "--is_need_transform", dest="need_transform", action="store_true"
    )
    parser.add_argument(
        "--no_need_transform", dest="need_transform", action="store_false"
    )

    if env == "CoatEnv-v0":
        parser.set_defaults(use_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int,
                            nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument("--leave_threshold", default=10, type=float)
        parser.add_argument("--num_leave_compute", default=3, type=int)
        parser.add_argument("--max_turn", default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "YahooEnv-v0":
        parser.set_defaults(use_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int,
                            nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument("--leave_threshold", default=120, type=float)
        parser.add_argument("--num_leave_compute", default=3, type=int)
        parser.add_argument("--max_turn", default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiRand-1K":
        parser.set_defaults(use_userinfo=False)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)

    elif env == "ml-1m":
        parser.set_defaults(use_userinfo=False)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=True)

    parser.add_argument("--force_length", type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)
    if env == "KuaiEnv-v0":
        args.use_userEmbedding = False

    return args


def get_datapath(envname):
    DATAPATH = None
    if envname == "CoatEnv-v0":
        DATAPATH = os.path.join(ROOTPATH, "environments", "coat")
    elif envname == "YahooEnv-v0":
        DATAPATH = os.path.join(ROOTPATH, "environments", "YahooR3")
    elif envname == "KuaiRand-1K":
        DATAPATH = os.path.join(
            ROOTPATH, "environments", "KuaiRand-1K", "data")
    elif envname == "ml-1m":
        DATAPATH = os.path.join(ROOTPATH, "environments", "ML_1M")
    return DATAPATH


def get_xy_columns(
    DataClass,
    df_data,
    df_user,
    df_item,
    user_features,
    item_features,
    reward_features,
    entity_dim,
    feature_dim,
):
    # if envname == "KuaiRand-1K":
    #     feat_item = [x for x in df_item.columns if x[:4] == "feat"]
    #     user_columns = \
    #         [SparseFeatP("user_id", df_data["user_id"].max() + 1, embedding_dim=entity_dim)] + \
    #         [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in user_features[1:]]
    #     item_columns = ([SparseFeatP("item_id", df_data["item_id"].max() + 1, embedding_dim=entity_dim)] + \
    #                     [SparseFeatP(x, df_item[feat_item].max().max() + 1, embedding_dim=feature_dim,
    #                                    embedding_name="feat",  # Share the same feature!
    #                                    padding_idx=0)  # using padding_idx in embedding!
    #                                    for x in feat_item
    #                     ] + \
    #                     [DenseFeat("duration_normed", 1)])
    #
    #     seq_columns = [VarLenSparseFeat(SparseFeat("hist_item", 3 + 1, embedding_dim=8), 4, length_name="seq_length")]
    #

    user_columns, item_columns, reward_columns = DataClass.get_xy_columns(df_user, df_item, user_features, item_features, reward_features, entity_dim, feature_dim)

    seq_columns = item_columns
    x_columns = user_columns + reward_columns
    y_column = item_columns[0]

    return x_columns, seq_columns, y_column


def get_DataClass(envname):
    if envname == "KuaiRand-1K":
        from environments.KuaiRand_1K.kuairand1k import KuaiRand1KEnv
        DataClass = KuaiRand1KEnv

    elif envname == "ml-1m":
        # from environments.ml1m import ML1MEnv, DATAPATH
        from environments.ML_1M.ml1m_data import ML1MData
        DataClass = ML1MData
    else:
        raise NotImplementedError
    return DataClass


def split_and_construct_dataset(df_user, df_item, x_columns, seq_columns, y_column, df_seq_rewards, hist_seq_dict, to_go_seq_dict, max_seq_length, len_reward_to_go):

    user_ids = df_seq_rewards["user_id"].to_numpy()
    session_start_id_list = [0] + list(np.diff(user_ids).nonzero()[0] + 1)
    session_end_id_list = list(np.array(session_start_id_list[1:]) - 1) + [len(user_ids) - 1]

    session_lens = np.array(session_end_id_list) - np.array(session_start_id_list) + 1
    assert session_lens.sum() == len(df_seq_rewards)
    test_session_idx = session_lens >= len_reward_to_go

    test_interaction_idx_left = (np.array(session_end_id_list) - len_reward_to_go + 1)[test_session_idx]
    test_interaction_idx_right = np.array(session_end_id_list)[test_session_idx]
    test_interaction_idx = np.concatenate([np.arange(x, y+1) for x, y in zip(test_interaction_idx_left, test_interaction_idx_right)])

    df_test_rewards = df_seq_rewards.loc[test_interaction_idx]
    test_hist_seq_dict = {key: item[test_interaction_idx] for key, item in hist_seq_dict.items()}
    test_to_go_seq_dict = {key: item[test_interaction_idx] for key, item in to_go_seq_dict.items()}

    print(1)
    train_idx = np.ones(len(df_seq_rewards), dtype=bool)
    train_idx[test_interaction_idx] = False
    # list_train_id = [x for x in df_seq_rewards.index if x not in test_interaction_idx]

    print(2)
    df_train_rewards = df_seq_rewards.loc[train_idx]
    print(3)
    train_hist_seq_dict = {key: item[train_idx]
                           for key, item in hist_seq_dict.items()}
    print(4)
    train_to_go_seq_dict = {
        key: item[train_idx] for key, item in to_go_seq_dict.items()
    }
    print(5)
    train_dataset = StateActionReturnDataset(
        df_user=df_user,
        df_item=df_item,
        x_columns=x_columns,
        y_column=y_column,
        seq_columns=seq_columns,
        max_seq_length=max_seq_length,
    )
    test_dataset = StateActionReturnDataset(
        df_user=df_user,
        df_item=df_item,
        x_columns=x_columns,
        y_column=y_column,
        seq_columns=seq_columns,
        max_seq_length=max_seq_length,
    )

    train_dataset.compile(df_train_rewards, train_hist_seq_dict, train_to_go_seq_dict)
    test_dataset.compile(df_test_rewards, test_hist_seq_dict, test_to_go_seq_dict)

    return train_dataset, test_dataset


def prepare_dataset(args):
    DataClass = get_DataClass(args.env)
    user_features, item_features, reward_features = DataClass.get_features(args.use_userinfo)

    df_data, df_user, df_item, list_feat = DataClass.get_data()

    x_columns, seq_columns, y_column = get_xy_columns(DataClass, df_data, df_user, df_item,
                                                      user_features, item_features, reward_features,
                                                      args.local_D, args.local_D)

    df_seq_rewards, hist_seq_dict, to_go_seq_dict = get_sequence_data(
        args.env, args.max_item_list_len, args.len_reward_to_go, args.reload
    )

    train_dataset, test_dataset = split_and_construct_dataset(df_user, df_item,
        x_columns, seq_columns, y_column, df_seq_rewards, hist_seq_dict, to_go_seq_dict, 
        args.max_item_list_len - args.len_reward_to_go, args.len_reward_to_go)

    EnvClass = DataClass.get_env_class()
    env = EnvClass(df_seq_rewards, target_features=reward_features, pareto_reload=args.reload)
    env.compile_test(df_data, df_user, df_item)
    mat = DataClass.get_completed_data()

    return train_dataset, test_dataset, env, mat


def get_sequence_data(envname, max_item_list_len, len_reward_to_go, reload):
    # df_train, df_user, df_item, list_feat = None, None, None, None
    DataClass = get_DataClass(envname)

    df_seq_rewards, hist_seq_dict, to_go_seq_dict = DataClass.get_seq_data(max_item_list_len, len_reward_to_go, reload=reload)

    return df_seq_rewards, hist_seq_dict, to_go_seq_dict


def get_features(envname, use_userinfo=False):
    if envname == "CoatEnv-v0":
        user_features = ["user_id", "gender_u", "age", "location", "fashioninterest"]
        item_features = ["item_id", "gender_i", "jackettype", "color", "onfrontpage"]
        reward_features = ["rating"]
    elif envname == "KuaiRand-1K":
        user_features = ["user_id", "user_active_degree", "is_live_streamer", "is_video_author", "follow_user_num_range",
                         "fans_user_num_range", "friend_user_num_range", "register_days_range",] + [f"onehot_feat{x}" for x in range(18)]
        if not use_userinfo:
            user_features = ["user_id"]
        item_features = (["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"])
        reward_features = ["is_click"]
    elif envname == "ml-1m":
        user_features = ["user_id"]
        if not use_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id", "rating"]
        reward_features = ["novelty", "diversity", "sum_rating"]
        # reward_features = ["novelty", "diversity", "sum_rating", "serendipity_4"]

    return user_features, item_features, reward_features