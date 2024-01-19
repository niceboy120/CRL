import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt

from environments.ML_1M import Movielens_MF_GroundTruth
from environments.ML_1M.ml1m_env import ML1MEnv
from environments.base_data import BaseData


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")


for path in [FIGPATH, RESULTPATH]:
    if not os.path.exists(path):
        os.mkdir(path)


class ML1MData(BaseData):

    def __init__(self, *args, **kwargs):
        self.ENVPATH = os.path.dirname(__file__)
        self.serendipity_threshold = [3,4,5]
        super().__init__(*args, **kwargs)
        self.item_padding_id = 0
        self.user_padding_id = 0

    @staticmethod
    def get_env_class():
        return ML1MEnv

    @staticmethod
    def get_features(df_user, df_item, use_userinfo=False):
        """
        Return: [Sparse features] and [Dense features]
        """
        user_features = {"sparse": ["user_id"],
                         "dense": []}
        if not use_userinfo:
            user_features = {"sparse": ["user_id"],
                             "dense": []}
        feat_item = [x for x in df_item.columns if x[:4] == "feat"]
        item_features = {"sparse": ["item_id"] + feat_item,
                         "dense": ["rating"]}
        # reward_features = ["novelty", "diversity", "sum_rating"] # Three metrics are hard to balance!
        reward_features = ["sum_rating", "diversity"]
        return user_features, item_features, reward_features


    def get_completed_data(self, device="cpu"):
        mat_path = os.path.join(RESULTPATH, "rating_matrix.csv")
        if os.path.exists(mat_path):
            mat = np.loadtxt(mat_path, delimiter=",")
        else:
            mat = Movielens_MF_GroundTruth.main(device=device)

        mat[mat < 0] = 0
        mat[mat > 5] = 5
        mat_new = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1))
        mat_new[1:, 1:] = mat
        return mat_new

    @staticmethod
    def load_user_feat():
        print("load user features")
        # filepath = os.path.join(DATAPATH, 'ml-1m.user')
        # df_user1 = pd.read_csv(filepath, sep="\t", header=0,
        #                        names=["user_id", "age", "gender", "occupation", "zip_code"],
        #                        dtype={0: int, 1: int, 2: str, 3: int, 4: str},)
        filepath = os.path.join(DATAPATH, 'users.dat')
        df_user = pd.read_csv(filepath, sep="::", header=None,
                              names=["user_id", "gender", "age", "occupation", "zip_code"],
                              dtype={0: int, 1: str, 2: int, 3: int, 4: str},
                              engine="python", )
        df_user["zip_code"] = df_user["zip_code"].apply(lambda x: x.split("-")[0])

        age_range = [0, 18, 25, 35, 45, 50, 56]
        df_user['age_range'] = pd.cut(df_user['age'], bins=age_range, labels=False)
        df_user['age_range'] += 1

        for col in ['gender', 'occupation', 'zip_code']:
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])

            df_user[col] += 1

        df_user.set_index("user_id", inplace=True)

        return df_user

    @staticmethod
    def load_category(tag_label="tags"):

        # filepath = os.path.join(DATAPATH, 'ml-1m.item')
        # df_item1 = pd.read_csv(filepath,
        #                       sep="\t",
        #                       header=0,
        #                       names=["item_id", "movie_title", "release_year", "genre"],
        #                       dtype={0: int, 1: str, 2: int, 3: str},)
        filepath = os.path.join(DATAPATH, 'movies.dat')
        df_item = pd.read_csv(filepath,
                              sep="::",
                              header=None,
                              names=["item_id", "movie_title and release_year", "genre"],
                              engine="python",
                              encoding="ISO-8859-1")  # 或尝试使用 encoding="Windows-1252"

        df_item["movie_title"] = df_item["movie_title and release_year"].apply(lambda x: x[:-7])
        df_item["release_year"] = df_item["movie_title and release_year"].apply(lambda x: x[-5:-1])
        df_item["release_year"] = df_item["release_year"].apply(lambda x: int(x) if x.isdigit() else 0)
        df_item.drop(columns=["movie_title and release_year"], inplace=True)

        df_item["genre"] = df_item["genre"].apply(lambda x: x.split("|"))  # using "|" to split the string
        df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))

        df_item.set_index("item_id", inplace=True)
        df_item_all = df_item.reindex(list(range(df_item.index.min(), df_item.index.max() + 1)))
        df_item_all.loc[df_item_all["genre"].isna(), "genre"] = df_item_all.loc[
            df_item_all["genre"].isna(), "genre"].apply(lambda x: [])

        list_feat = df_item_all['genre'].to_list()

        # df_item["year_range"] = pd.cut(df_item["release_year"], bins=[1900, 1950, 2000, 2050], labels=False)

        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
                               index=df_item_all.index)

        lbe = LabelEncoder()
        tags_array = lbe.fit_transform(df_feat.to_numpy().reshape(-1)).reshape(df_feat.shape)
        tags_array += 1
        tags_array[tags_array.max() == tags_array] = 0
        df_feat = pd.DataFrame(tags_array, columns=df_feat.columns, index=df_feat.index)

        list_feat_num = list(map(lambda x: lbe.transform(x) + 1, list_feat))
        df_feat[tag_label] = list_feat_num

        return list_feat_num, df_feat

    @staticmethod
    def load_item_feat():
        list_feat, df_feat = ML1MData.load_category()
        df_item = df_feat
        return df_item



    @staticmethod
    def get_df_ml_1m(inter_path):
        # df = pd.read_csv(
        #     inter_path,
        #     sep="\t",
        #     header=0,
        #     dtype={0: int, 1: int, 2: float, 3: int},
        #     names=["user_id", "item_id", "rating", "timestamp"],
        # )
        df = pd.read_csv(
            inter_path,
            sep="::",
            header=None,
            dtype={0: int, 1: int, 2: float, 3: int},
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        print(df)

        df.sort_values(by=["user_id", "timestamp"], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # transfer timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")

        # transfer datetime to date
        df["day"] = df["date"].dt.date

        df.groupby(["user_id", "day"]).agg(len).describe()
        df.groupby(["user_id"]).agg(len).describe()

        return df


    @staticmethod
    def data_augment(df_data, augment_rate=1):
        from tqdm import tqdm
        if augment_rate <= 0:
            return df_data
        all_user_id = df_data['user_id'].unique()
        auxiliary_data = []
        for uid in tqdm(all_user_id, desc="Iterating for augmentation"):
            user_interactions = df_data[df_data['user_id'] == uid]
            user_item_num = len(user_interactions)
            sampled = user_interactions.sample(n=int(user_item_num * augment_rate), replace=True, random_state=42)
            # set sampled's timestamp to 0 (so that it will always appears before real interactions)
            sampled['timestamp'] = 0
            sampled['date'] = pd.to_datetime(sampled['timestamp'], unit='s')
            sampled['day'] = sampled['date'].dt.date
            auxiliary_data.append(sampled)
        auxiliary_df = pd.concat(auxiliary_data)
        # add to original data
        augmented_df = pd.concat([df_data, auxiliary_df])
        augmented_df.sort_values(by=["user_id", "timestamp"], ascending=True, inplace=True)
        return augmented_df


    # @staticmethod
    def get_data(self):
        inter_path = os.path.join(DATAPATH, "ratings.dat")
        user_feat_path = os.path.join(DATAPATH, "users.dat")
        item_feat_path = os.path.join(DATAPATH, "movies.dat")

        df = ML1MData.get_df_ml_1m(inter_path)

        # len_inters = [len(x[1]) for x in df.groupby("user_id")]
        # len_inters = np.array(len_inters)
        # np.percentile(len_inters, 20)

        df_user = ML1MData.load_user_feat()
        # df_item = ML1MEnv.load_item_feat()
        list_feat, df_item = ML1MData.load_category()

        return df, df_user, df_item, list_feat




def visualize_3d(seq_df):
    df_visual = seq_df[["novelty", "diversity", "serendipity"]]

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

    plt.savefig(os.path.join(FIGPATH, "3d_scatter.pdf"), bbox_inches="tight", pad_inches=0)

    plt.show()


def visualize_2d(seq_df):
    df_visual = seq_df[["novelty", "diversity", "serendipity", "sum_rating"]].iloc[:10000]

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

    plt.savefig(os.path.join(FIGPATH, "2d_scatters.pdf"), bbox_inches="tight", pad_inches=0)

    plt.close()


def get_pairgrid_plot(seq_df):
    df_visual = seq_df[["novelty", "diversity", "serendipity", "sum_rating"]].iloc[:10000]

    g = sns.PairGrid(df_visual, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    plt.savefig(os.path.join(FIGPATH, "pairgrid_plot.pdf"), bbox_inches="tight", pad_inches=0)
    plt.close()

    # seq_df["serendipity"] = seq_df["serendipity_5"]
    # get_pairgrid_plot(seq_df)


if __name__ == "__main__":
    pass
