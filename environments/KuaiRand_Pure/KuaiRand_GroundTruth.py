import argparse
import torch


from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM, WDL

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data")
RESULTPATH = os.path.join(CODEPATH, "data_processed")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_dim", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cuda", type=int, default=7)
    parser.add_argument("--message", type=str, default="debug")
    parser.add_argument("--hidden_size", type=int, nargs='+', default=(32, 32))
    args = parser.parse_known_args()[0]
    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{args.cuda}'
    args.device = device
    print(args)
    return args



def get_features(df_user, df_item):
    """
    Return: [Sparse features] and [Dense features]
    """
    user_features = {"sparse": ["user_id"],
                     "dense": []}
    item_features = {"sparse": ["item_id"],
                     "dense": []
                     }

    # reward_features = ["novelty", "diversity", "sum_rating"] # Three metrics are hard to balance!
    feedback_dense_features = ["rating"]
    return user_features, item_features, feedback_dense_features



def get_xy_columns(df_user, df_item, user_features, item_features, feedback_dense_features, entity_dim, feature_dim):
    feat_user = [x for x in user_features["sparse"] if x[:5] == "ufeat"]
    nofeat_user = [x for x in user_features["sparse"] if x[:5] != "ufeat" and x != df_user.index.name]
    user_columns = \
        [SparseFeat("user_id", df_user.index.max() + 1, embedding_dim=entity_dim)] + \
        [SparseFeat(col, df_user[col].max() + 1, embedding_dim=feature_dim, ) for col in nofeat_user] + \
        [SparseFeat(x, df_user[feat_user].max().max() + 1, embedding_dim=feature_dim,
                     embedding_name="ufeat",  # Share the same feature!
                     ) for x in feat_user] + \
        [DenseFeat(col, dimension=1, ) for col in user_features["dense"]]

    feat_item = [x for x in item_features["sparse"] if x[:4] == "feat"]
    nofeat_item = [x for x in item_features["sparse"] if x[:4] != "feat" and x != df_item.index.name]
    item_columns = \
        [SparseFeat("item_id", df_item.index.max() + 1, embedding_dim=entity_dim)] + \
        [SparseFeat(col, df_item[col].max() + 1, embedding_dim=feature_dim, ) for col in nofeat_item] + \
        [SparseFeat(x, df_item[feat_item].max().max() + 1, embedding_dim=feature_dim,
                     embedding_name="feat",  # Share the same feature!
                     ) for x in feat_item] + \
        [DenseFeat(col, dimension=1, ) for col in item_features["dense"]]
    reward_columns = [DenseFeat(name, dimension=1, ) for name in feedback_dense_features]

    return user_columns, item_columns, reward_columns






def get_feat(df_entity, topic_col, tag_label="tags", is_user=False):
    df_entity.loc[df_entity[topic_col].isna(), topic_col] = df_entity.loc[
        df_entity[topic_col].isna(), topic_col].apply(lambda x: [])
    list_feat_str = df_entity[topic_col].to_list()
    list_feat = list(map(lambda x: x.split(",") if type(x) is str else [], list_feat_str))
    list_feat_num = list(map(lambda x: np.array([int(y) + 1 for y in x]), list_feat))  # Note: +1 for padding_idx=0

    max_topic = max(map(len, list_feat_num))
    feat_name = "ufeat" if is_user else "feat"
    df_feat = pd.DataFrame(list_feat_num, columns=[f'{feat_name}{i}' for i in range(max_topic)], index=df_entity.index)
    df_feat += 1  # Note: +1 for padding_idx=0
    df_feat.fillna(0, inplace=True)
    df_feat = df_feat.astype(int)

    df_feat[tag_label] = list_feat_num
    df_entity = df_entity.join(df_feat)
    return df_entity, list_feat_num


def load_user_feat():
    print("load user features")
    filepath = os.path.join(DATAPATH, 'user_features_pure.csv')
    df_user = pd.read_csv(filepath, header=0)

    features_to_encode = ["user_active_degree", "follow_user_num_range", "fans_user_num_range"]
    # df_user.loc[df_user["user_active_degree"] == "UNKNOWN", "user_active_degree"] = np.nan
    # df_user.loc[df_user["follow_user_num_range"] == "UNKNOWN", "user_active_degree"] = np.nan
    for feat in features_to_encode:
        lbe = LabelEncoder()
        lbe.fit_transform(df_user[feat])
        df_user[feat] = lbe.transform(df_user[feat])
        df_user[feat] += 1
        if type(lbe.classes_[-1]) is float and np.isnan(lbe.classes_[-1]):
            df_user.loc[df_user[feat] == df_user[feat].max(), feat] = 0

    df_user.set_index("user_id", inplace=True)

    return df_user


def get_df_kuairand_pure():
    inter_path = os.path.join(DATAPATH, "log_standard_4_08_to_4_21_pure.csv")
    df = pd.read_csv(inter_path, header=0)

    df.rename(columns={"video_id": "item_id"}, inplace=True)
    df["rating"] = df["is_click"]
    df.sort_values(by=["user_id", "time_ms"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def load_category(tag_label="tags"):
    filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
    df_item = pd.read_csv(filepath, header=0)
    df_item.rename(columns={"video_id": "item_id"}, inplace=True)

    df_item.set_index("item_id", inplace=True)

    df_item_feat, list_feat_num = get_feat(df_item, "tag", tag_label=tag_label)

    return list_feat_num, df_item_feat


def load_item_feat(only_small=False):
    list_feat, df_feat = load_category()
    df_item = df_feat
    return df_item


def get_data(reserved_items=5000, reserved_users=10000):
    df = get_df_kuairand_pure()
    df = df.loc[df["item_id"] < reserved_items]
    df = df.loc[df["user_id"] < reserved_users]

    # len_inters = [len(x[1]) for x in df.groupby("user_id")]
    # len_inters = np.array(len_inters)
    # np.percentile(len_inters, 20)

    df_user = load_user_feat()
    df_user = df_user.loc[df_user.index < reserved_users]

    # df_item = ML1MEnv.load_item_feat()
    list_feat, df_item = load_category()
    df_item = df_item.loc[df_item.index < reserved_items]

    print(df)

    return df, df_user, df_item, list_feat





def main(device="cpu"):
    args = get_args()
    df, df_user, df_item, list_feat = get_data()

    user_features, item_features, feedback_dense_features = get_features(df_user, df_item)
    user_columns, item_columns, reward_columns = get_xy_columns(
        df_user, df_item, user_features, item_features, feedback_dense_features, entity_dim=args.entity_dim, feature_dim=args.feature_dim)

    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed)
    # df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=args.seed)

    df_user_part = df_user.reset_index()[
        [col.name for col in user_columns if col.name in df_user.reset_index()]].set_index("user_id")
    df_item_part = df_item.reset_index()[
        [col.name for col in item_columns if col.name in df_item.reset_index()]].set_index("item_id")
    df_data = df.join(df_user_part, on="user_id", how="left")
    df_data = df_data.join(df_item_part, on="item_id", how="left")


    # df_pos = df_data.loc[df_data["rating"] == 1]
    # df_data.loc[df_data["rating"] == 0].index
    # sampling_num = int(df_data["rating"].sum())
    # df_neg = df_data.loc[df_data["rating"] == 0].sample(sampling_num, random_state=args.seed, replace=False)
    #
    # df_data_new = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_data_new = df_data.sample(frac=1, random_state=args.seed, replace=False).reset_index(drop=True)

    x_list = [df_data_new[column.name].to_numpy() for column in user_columns + item_columns]
    y_list = [df_data_new[column.name].to_numpy() for column in reward_columns]

    # balance label 0 and 1 in y_list by resampling:

    model = WDL(user_columns + item_columns, user_columns + item_columns, dnn_activation='prelu',
                dnn_hidden_units=args.hidden_size, dnn_dropout=0.3, device=device)

    # model = DeepFM(user_columns + item_columns, user_columns + item_columns, use_fm=True,
    #                dnn_hidden_units=args.hidden_size, dnn_dropout=0.5, device=device)

    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy', 'acc', 'mae'])
    model.fit(x_list, y_list, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1)


    # Predict the entire rating matrix
    num_users = user_columns[user_features["sparse"].index("user_id")].vocabulary_size
    num_items = item_columns[item_features["sparse"].index("item_id")].vocabulary_size


    all_users = torch.tensor(range(num_users), dtype=torch.int64)
    all_items = torch.tensor(range(num_items), dtype=torch.int64)


    # Use torch.cartesian_prod to generate all user-item pairs
    all_user_item_pairs = torch.cartesian_prod(all_users, all_items)
    all_user_item_pairs = [all_user_item_pairs[:, i] for i in range(all_user_item_pairs.shape[1])]

    # all_users = np.arange(num_users - 1, dtype=np.int64)
    # all_items = np.arange(num_items - 1, dtype=np.int64)
    #
    # # Generate all user-item pairs using np.meshgrid
    # all_user_item_pairs = np.array(np.meshgrid(all_users, all_items)).T.reshape(-1, 2)
    # all_user_item_pairs_numpy = all_user_item_pairs.cpu().numpy()

    model.eval()  # Set the model to evaluation mode
    print("predicting...")
    with torch.no_grad():
        pred_ans = model.predict(all_user_item_pairs, args.batch_size * 20)

    rating_matrix = pred_ans.reshape(num_users, num_items)

    rating_matrix_save_path = os.path.join(RESULTPATH, 'rating_matrix.csv')
    np.savetxt(rating_matrix_save_path, rating_matrix, delimiter=',')
    print("results saved.")
    return rating_matrix


    # print(model_name + 'test, train valid pass!')





if __name__ == "__main__":
    main()