import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

from environments.KuaiRand_Pure import KuaiRand_GroundTruth
from environments.KuaiRand_Pure.kuairand_pure_env import KuaiRandPureEnv
from environments.base_data import BaseData


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")

FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")


for path in [FIGPATH, RESULTPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class KuaiRandPureData(BaseData):
    def __init__(self, *args, **kwargs):
        self.ENVPATH = os.path.dirname(__file__)
        self.serendipity_threshold = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_env_class():
        return KuaiRandPureEnv

    @staticmethod
    def get_features(df_user, df_item, use_userinfo=False):
        """
        Return: [Sparse features] and [Dense features]
        """
        user_features = {"sparse":
                             ["user_id", "user_active_degree", "follow_user_num_range", "fans_user_num_range"],
                         "dense":
                             []}
        if not use_userinfo:
            user_features = {"sparse":["user_id"], "dense":[]}
        # item_features = ["item_id", "click"]

        feat_item = [x for x in df_item.columns if x[:4] == "feat"]
        item_features = {"sparse": ['item_id'] + feat_item,
                         "dense": ["rating"]
                         }
        # reward_features = ["novelty", "diversity", "sum_rating"] # Three metrics are hard to balance!
        reward_dense_features = ["sum_rating", "diversity"]
        return user_features, item_features, reward_dense_features

    def get_completed_data(self, device="cpu"):
        mat_path = os.path.join(RESULTPATH, "rating_matrix.csv")
        if os.path.exists(mat_path):
            mat = np.loadtxt(mat_path, delimiter=",")
        else:
            mat = KuaiRand_GroundTruth.main(device)


        return mat

    @staticmethod
    def load_user_feat(tag_label="utags"):
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

    @staticmethod
    def load_category(tag_label="tags"):
        filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
        df_item = pd.read_csv(filepath, header=0)
        df_item.rename(columns={"video_id": "item_id"}, inplace=True)

        df_item.set_index("item_id", inplace=True)

        df_item_feat, list_feat_num = get_feat(df_item, "tag", tag_label=tag_label)

        return list_feat_num, df_item_feat

    @staticmethod
    def load_item_feat():
        list_feat, df_feat = KuaiRandPureData.load_category()
        df_item = df_feat
        return df_item


    @staticmethod
    def get_df_kuairand_pure():
        inter_path = os.path.join(DATAPATH, "log_standard_4_08_to_4_21_pure.csv")
        df = pd.read_csv(inter_path, header=0)

        df.rename(columns={"video_id": "item_id"}, inplace=True)
        df["rating"] = df["is_click"]
        df.sort_values(by=["user_id", "time_ms"], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

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
            # set sampled's time to 0 (so that it will always appears before real interactions)
            sampled['time_ms'] = 0
            sampled['date'] = pd.to_datetime(sampled['time_ms'], unit='ms')
            auxiliary_data.append(sampled)
        auxiliary_df = pd.concat(auxiliary_data)
        # add to original data
        augmented_df = pd.concat([df_data, auxiliary_df])
        augmented_df.sort_values(by=["user_id", "time_ms"], ascending=True, inplace=True)
        return augmented_df


    def get_data(self, reserved_items=5000, reserved_users=10000):
        df = KuaiRandPureData.get_df_kuairand_pure()
        df = df.loc[df["item_id"] < reserved_items]
        df = df.loc[df["user_id"] < reserved_users]

        # len_inters = [len(x[1]) for x in df.groupby("user_id")]
        # len_inters = np.array(len_inters)
        # np.percentile(len_inters, 20)

        df_user = KuaiRandPureData.load_user_feat()
        df_user = df_user.loc[df_user.index < reserved_users]

        # df_item = ML1MEnv.load_item_feat()
        list_feat, df_item = KuaiRandPureData.load_category()
        df_item = df_item.loc[df_item.index < reserved_items]

        print(df)

        return df, df_user, df_item, list_feat

    # @staticmethod



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


def main():
    pass

if __name__ == "__main__":
    main()
