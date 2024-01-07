import numpy as np
import pandas as pd
import os


from environments.Zhihu_1M import Zhihu_WDL_GroundTruth
from environments.Zhihu_1M.zhihu1m_env import Zhihu1MEnv
from environments.base_data import BaseData


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")


for path in [FIGPATH, RESULTPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class Zhihu1MData(BaseData):
    def __init__(self, *args, **kwargs):
        self.ENVPATH = os.path.dirname(__file__)
        self.serendipity_threshold = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_env_class():
        return Zhihu1MEnv

    @staticmethod
    def get_features(df_user, df_item, use_userinfo=False):
        """
        Return: [Sparse features] and [Dense features]
        """
        user_features = {"sparse":
                             ["user_id", "gender"],
                         "dense":
                             ["num_followers", "num_topics_followed", "num_questions_followed",
                              "num_answers", "num_questions", "num_comments", "num_thanks_received",
                              "num_comments_received", "num_likes_received", "num_dislikes_received"]}
        if not use_userinfo:
            user_features = {"sparse":["user_id"], "dense":[]}
        # item_features = ["item_id", "click"]
        item_features = {"sparse": ['item_id'] + ["is_high_value", "recommended_by_the_editor_or_not", "contain_pictures", "contain_videos"],
                         "dense": ["num_thanks", "num_likes", "num_comments", "num_collections", "num_dislikes", "num_reported", "num_helpless", "rating"]
                         }

        # reward_features = ["novelty", "diversity", "sum_rating"] # Three metrics are hard to balance!
        reward_dense_features = ["sum_rating", "diversity"]
        return user_features, item_features, reward_dense_features

    def get_completed_data(self, device="cpu"):
        mat_path = os.path.join(RESULTPATH, "rating_matrix.csv")
        if os.path.exists(mat_path):
            mat = np.loadtxt(mat_path, delimiter=",")
        else:
            mat = Zhihu_WDL_GroundTruth.main(device=device)
        return mat

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def load_item_feat(only_small=False):
        list_feat, df_feat = Zhihu1MData.load_category()
        df_item = df_feat
        return df_item


    @staticmethod
    def get_df_zhihu_1m():
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


    @staticmethod
    def data_augment(df_data, k=1):
        from tqdm import tqdm
        if k <= 0:
            return df_data
        all_user_id = df_data['user_id'].unique()
        auxiliary_data = []
        for uid in tqdm(all_user_id, desc="Iterating for augmentation"):
            user_interactions = df_data[df_data['user_id'] == uid]
            user_item_num = len(user_interactions)
            sampled = user_interactions.sample(n=int(user_item_num*k), replace=True, random_state=42)
            # set sampled's timestamp to 0 (so that it will always appears before real interactions)
            sampled['impression timestamp'] = 0
            sampled['date'] = pd.to_datetime(sampled['impression timestamp'], unit='s')
            sampled['day'] = sampled['date'].dt.date
            auxiliary_data.append(sampled)
        auxiliary_df = pd.concat(auxiliary_data)
        # add to original data
        augmented_df = pd.concat([df_data, auxiliary_df])
        augmented_df.sort_values(by=["user_id", "impression timestamp"], ascending=True, inplace=True)
        return augmented_df


    def get_data(self, reserved_items=5000, reserved_users=10000):

        df = Zhihu1MData.get_df_zhihu_1m()
        df = df.loc[df["item_id"] < reserved_items]
        df = df.loc[df["user_id"] < reserved_users]

        # len_inters = [len(x[1]) for x in df.groupby("user_id")]
        # len_inters = np.array(len_inters)
        # np.percentile(len_inters, 20)

        df_user = Zhihu1MData.load_user_feat()
        df_user = df_user.loc[df_user.index < reserved_users]

        # df_item = ML1MEnv.load_item_feat()
        list_feat, df_item = Zhihu1MData.load_category()
        df_item = df_item.loc[df_item.index < reserved_items]

        print(df)

        return df, df_user, df_item, list_feat

    # @staticmethod



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


def load_data():
    # load dataset from files:
    # 读取 inter_impression.csv
    df_inter_impression = pd.read_csv(os.path.join(DATAPATH, "inter_impression.csv"), header=None,
                                      names=["user_id", "answer_id", "impression timestamp", "click timestamp"])

    # 读取 info_user.csv
    df_info_user = pd.read_csv(os.path.join(DATAPATH, "info_user.csv"), header=None, sep=",",
                               names=["user_id", "register_timestamp", "gender", "login_frequency",
                                      "num_followers", "num_topics_followed", "num_questions_followed",
                                      "num_answers", "num_questions", "num_comments", "num_thanks_received",
                                      "num_comments_received", "num_likes_received", "num_dislikes_received",
                                      "register_type", "register_platform", "from_android", "from_iphone",
                                      "from_ipad", "from_pc", "from_mobile_web", "device_model", "device_brand",
                                      "platform", "province", "city", "topic_IDs_followed"])

    # 读取 inter_query.csv
    df_inter_query = pd.read_csv(os.path.join(DATAPATH, "inter_query.csv"), header=None,
                                 names=["user_id", "token_id", "query_timestamp"])

    # 读取 info_answer.csv
    df_info_answer = pd.read_csv(os.path.join(DATAPATH, "info_answer.csv"), header=None, sep=",",
                                 names=["answer_id", "question_id", "is_anonymous", "author_id", "is_high_value",
                                        "recommended_by_the_editor_or_not", "creation_timestamp", "contain_pictures",
                                        "contain_videos", "num_thanks", "num_likes", "num_comments", "num_collections",
                                        "num_dislikes", "num_reported", "num_helpless", "token_IDs", "topic_IDs"])

    # 读取 info_question.csv
    df_info_question = pd.read_csv(os.path.join(DATAPATH, "info_question.csv"), header=None, sep=",",
                                   names=["question_id", "timestamp", "num_answers", "num_followers", "num_invitations",
                                          "num_comments", "token_IDs", "topic_IDs"])

    # 读取 info_author.csv
    df_info_author = pd.read_csv(os.path.join(DATAPATH, "info_author.csv"), header=None,
                                 names=["author_id", "is_excellent_author", "num_followers", "is_excellent_answerer"])

    # 读取 info_topic.csv
    df_info_topic = pd.read_csv(os.path.join(DATAPATH, "info_topic.csv"), header=None, names=["topic_id"])

    return df_inter_impression, df_info_user, df_inter_query, df_info_answer, df_info_question, df_info_author, df_info_topic


def main():
    df_inter_impression, df_info_user, df_inter_query, df_info_answer, df_info_question, df_info_author, df_info_topic = load_data()
    df_inter_impression["answer_id"].nunique()
    df_info_answer["answer_id"].nunique()
    df_info_answer["answer_id"].nunique()


if __name__ == "__main__":
    main()
