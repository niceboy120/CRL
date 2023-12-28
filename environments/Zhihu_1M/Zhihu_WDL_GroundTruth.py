import argparse
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import WDL
import os
import pandas as pd
import numpy as np



CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data")
RESULTPATH = os.path.join(CODEPATH, "data_processed")


# write a args parser:
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_dim", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cuda", type=int, default=2)
    parser.add_argument("--message", type=str, default="debug")
    parser.add_argument("--hidden_size", type=int, nargs='+', default=(64, 64))
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
    # user_features = {"sparse":
    #                      ["user_id", "gender", "from_android", "from_iphone", "from_ipad", "from_pc", "from_mobile_web"] + [f"ufeat{i}" for i in range(10)],
    #                  "dense":
    #                      ["num_followers", "num_topics_followed", "num_questions_followed",
    #                       "num_answers", "num_questions", "num_comments", "num_thanks_received",
    #                       "num_comments_received", "num_likes_received", "num_dislikes_received"]}
    # # item_features = ["item_id", "click"]
    # item_features = {"sparse": ['item_id'] + ["is_high_value", "recommended_by_the_editor_or_not", "contain_pictures", "contain_videos"] + [f"feat{i}" for i in range(11)],
    #                  "dense": ["num_thanks", "num_likes", "num_comments", "num_collections", "num_dislikes", "num_reported", "num_helpless"]
    #                  }

    # user_features = {"sparse":
    #                      ["user_id", "gender", "from_android", "from_iphone", "from_ipad", "from_pc", "from_mobile_web"],
    #                  "dense":
    #                      []
    #                  }
    # item_features = {"sparse": ['item_id'] + ["is_high_value", "recommended_by_the_editor_or_not", "contain_pictures", "contain_videos"],
    #                  "dense": []
    #                  }

    # user_features = {"sparse":
    #                      ["user_id"] ,
    #                  "dense":
    #                      []
    #                      }
    # item_features = {"sparse": ['item_id'] + ["is_high_value", "recommended_by_the_editor_or_not", "contain_pictures", "contain_videos"] + [f"feat{i}" for i in range(11)],
    #                  "dense": []
    #                  }

    # user_features = {"sparse":["user_id"],
    #                  "dense":[]}
    # # item_features = ["item_id", "click"]
    # item_features = {"sparse": ['item_id'] + ["is_high_value", "recommended_by_the_editor_or_not", "contain_pictures", "contain_videos"],
    #                  "dense": []
    #                  }
    #
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
    list_feat = list(map(lambda x: x.split() if type(x) is str else [], list_feat_str))
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

def load_user_feat(tag_label="utags"):
    print("load user features")
    filepath = os.path.join(DATAPATH, 'info_user.csv')
    df_user = pd.read_csv(os.path.join(DATAPATH, "info_user.csv"), header=None, sep=",",
                          names=["user_id", "register_timestamp", "gender", "login_frequency",
                                 "num_followers", "num_topics_followed", "num_questions_followed",
                                 "num_answers", "num_questions", "num_comments_user", "num_thanks_received",
                                 "num_comments_received", "num_likes_received", "num_dislikes_received",
                                 "register_type", "register_platform", "from_android", "from_iphone",
                                 "from_ipad", "from_pc", "from_mobile_web", "device_model", "device_brand",
                                 "platform", "province", "city", "topic_IDs_followed"])

    df_user.set_index("user_id", inplace=True)

    df_user_new, list_feat_num = get_feat(df_user, "topic_IDs_followed", tag_label=tag_label, is_user=True)

    return df_user_new

def get_df_zhihu_1m(inter_path):
    df = pd.read_csv(os.path.join(DATAPATH, "inter_impression.csv"), header=None,
                     names=["user_id", "item_id", "impression timestamp", "click timestamp"])


    # Define rating for zhihu dataset. 0 for impression, 1 for click.
    df["rating"] = 0.0
    df.loc[df["click timestamp"] != 0, "rating"] = 1.0

    df.sort_values(by=["user_id", "impression timestamp"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # transfer timestamp to datetime
    df["date"] = pd.to_datetime(df["impression timestamp"], unit="s")

    # transfer datetime to date
    df["day"] = df["date"].dt.date

    df.groupby(["user_id", "day"]).agg(len).describe()
    df.groupby(["user_id"]).agg(len).describe()

    return df


def load_category(tag_label="tags"):
    filepath = os.path.join(DATAPATH, 'info_answer.csv')
    df_item = pd.read_csv(os.path.join(DATAPATH, "info_answer.csv"), header=None, sep=",",
                          names=["item_id", "question_id", "is_anonymous", "author_id", "is_high_value",
                                 "recommended_by_the_editor_or_not", "creation_timestamp",
                                 "contain_pictures",
                                 "contain_videos", "num_thanks", "num_likes", "num_comments",
                                 "num_collections",
                                 "num_dislikes", "num_reported", "num_helpless", "token_IDs", "topic_IDs"])

    df_item.set_index("item_id", inplace=True)

    df_item, list_feat_num = get_feat(df_item, "topic_IDs", tag_label=tag_label)

    return list_feat_num, df_item


def load_item_feat(only_small=False):
    list_feat, df_feat = load_category()
    df_item = df_feat
    return df_item

def get_data(reserved_items=5000, reserved_users=10000):
    inter_path = os.path.join(DATAPATH, "inter_impression.csv")

    df = get_df_zhihu_1m(inter_path)

    df = df.loc[df["item_id"] < reserved_items]
    df = df.loc[df["user_id"] < reserved_users]


    df_user = load_user_feat()
    # df_item = ML1MEnv.load_item_feat()
    list_feat, df_item = load_category()

    df_item = df_item.loc[df_item.index < reserved_items]
    df_user = df_user.loc[df_user.index < reserved_users]

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


    df_pos = df_data.loc[df_data["rating"] == 1]
    df_data.loc[df_data["rating"] == 0].index
    sampling_num = int(df_data["rating"].sum())
    df_neg = df_data.loc[df_data["rating"] == 0].sample(sampling_num, random_state=args.seed, replace=False)

    df_data_new = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    x_list = [df_data_new[column.name].to_numpy() for column in user_columns + item_columns]
    y_list = [df_data_new[column.name].to_numpy() for column in reward_columns]

    # balance label 0 and 1 in y_list by resampling:

    model = WDL(user_columns + item_columns, user_columns + item_columns, dnn_activation='prelu',
                dnn_hidden_units=args.hidden_size, dnn_dropout=0.1, device=device)

    # model = DeepFM(user_columns + item_columns, user_columns + item_columns, use_fm=True,
    #                dnn_hidden_units=args.hidden_size, dnn_dropout=0.3, device=args.device)

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