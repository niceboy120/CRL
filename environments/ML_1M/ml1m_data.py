import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

from environments.ML_1M.ml1m_env import ML1MEnv
from environments.base_data import BaseData
from inputs import SparseFeatP, DenseFeat
from ..utils import get_novelty, get_serendipity, get_diversiy


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")

seq_columns = ['item_id'] + [f"feat{x}" for x in range(6)] + ["rating"]
# seq_columns = ['item_id'] + [f"feat{x}" for x in range(6)]



for path in [FIGPATH, RESULTPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class ML1MData(BaseData):

    @staticmethod
    def get_env_class():
        return ML1MEnv

    @staticmethod
    def get_xy_columns(df_user, df_item, user_features, item_features, reward_features, entity_dim, feature_dim):
        feat_item = [x for x in df_item.columns if x[:4] == "feat"]
        user_columns = \
            [SparseFeatP("user_id", df_user.index.max() + 1, embedding_dim=entity_dim)] + \
            [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in user_features[1:]]
        item_columns = \
            [SparseFeatP("item_id", df_item.index.max() + 1, embedding_dim=entity_dim)] + \
            [SparseFeatP(x, df_item[feat_item].max().max() + 1, embedding_dim=feature_dim,
                         embedding_name="feat",  # Share the same feature!
                         padding_idx=0,  # using padding_idx in embedding!
                         ) for x in feat_item] + \
            [DenseFeat("rating", dimension=1, embedding_dim=feature_dim)]
        reward_columns = [DenseFeat(name, dimension=1, embedding_dim=feature_dim) for name in reward_features]
        return user_columns, item_columns, reward_columns

    @staticmethod
    def get_features(use_userinfo=False):
        user_features = ["user_id"]
        if not use_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id", "rating"]
        reward_features = ["novelty", "diversity", "sum_rating"]
        return user_features, item_features, reward_features
    
    @staticmethod
    def get_completed_data():
        mat_path = os.path.join(DATAPATH, "rating_matrix.csv")
        mat = np.loadtxt(mat_path, delimiter=",")
        
        mat[mat<0] = 0
        mat[mat > 5] = 5
        mat_new = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1))
        mat_new[1:, 1:] = mat
        return mat_new

    @staticmethod
    def load_user_feat():
        print("load user features")
        filepath = os.path.join(DATAPATH, 'ml-1m.user')
        df_user = pd.read_csv(filepath, sep="\t", header=0,
                               names=["user_id", "age", "gender", "occupation", "zip_code"],
                               dtype={0: int, 1: int, 2: str, 3: int, 4: str},)
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
        
        filepath = os.path.join(DATAPATH, 'ml-1m.item')
        df_item = pd.read_csv(filepath, 
                              sep="\t", 
                              header=0,
                              names=["item_id", "movie_title", "release_year", "genre"],
                              dtype={0: int, 1: str, 2: int, 3: str},)
        df_item["genre"] = df_item["genre"].apply(lambda x: x.split())
        df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))

        
        df_item.set_index("item_id", inplace=True)
        df_item_all = df_item.reindex(list(range(df_item.index.min(),df_item.index.max()+1)))
        df_item_all.loc[df_item_all["genre"].isna(), "genre"] = df_item_all.loc[df_item_all["genre"].isna(), "genre"].apply(lambda x: [])

        list_feat = df_item_all['genre'].to_list()

        # df_item["year_range"] = pd.cut(df_item["release_year"], bins=[1900, 1950, 2000, 2050], labels=False)


        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3', 'feat4', 'feat5'], index=df_item_all.index)
        
        lbe = LabelEncoder()
        tags_array = lbe.fit_transform(df_feat.to_numpy().reshape(-1)).reshape(df_feat.shape)
        tags_array += 1
        tags_array[tags_array.max() == tags_array] = 0
        df_feat = pd.DataFrame(tags_array, columns=df_feat.columns, index=df_feat.index)

        list_feat_num = list(map(lambda x: lbe.transform(x) + 1, list_feat))
        df_feat[tag_label] = list_feat_num
        
        return list_feat_num, df_feat

    @staticmethod
    def load_item_feat(only_small=False):
        list_feat, df_feat = ML1MData.load_category()
        df_item = df_feat
        return df_item
    
    @staticmethod
    def get_seq_data(max_item_list_len, len_reward_to_go, reload):
        # columns = ['user_id', 'item_id', 'rating', "timestamp"]
        df_seq_rewards, hist_seq_dict, to_go_seq_dict = BaseData.get_and_save_seq_data(ML1MData, RESULTPATH, seq_columns, max_item_list_len, len_reward_to_go, reload=reload)
        return df_seq_rewards, hist_seq_dict, to_go_seq_dict

    @staticmethod
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
        df.reset_index(drop=True, inplace=True)

        # transfer timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")

        # transfer datetime to date
        df["day"] = df["date"].dt.date

        df.groupby(["user_id", "day"]).agg(len).describe()
        df.groupby(["user_id"]).agg(len).describe()

        return df

    @staticmethod
    def get_data():
        inter_path = os.path.join(DATAPATH, "ml-1m.inter")
        user_feat_path = os.path.join(DATAPATH, "ml-1m.user")
        item_feat_path = os.path.join(DATAPATH, "ml-1m.item")

        df = ML1MData.get_df_ml_1m(inter_path)
        
        # len_inters = [len(x[1]) for x in df.groupby("user_id")]
        # len_inters = np.array(len_inters)
        # np.percentile(len_inters, 20)


        df_user = ML1MData.load_user_feat()
        # df_item = ML1MEnv.load_item_feat()
        list_feat, df_item = ML1MData.load_category()

        return df, df_user, df_item, list_feat

    @staticmethod
    def get_statistics(df, df_user, df_item, target_df, to_go_seq_dict, tag_label = "tags"):
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
        target_df.loc[:,"novelty"] = get_novelty(item_novelty, to_go_item_array)

        # item_index_dict = dict(
        #     zip(df_item.index.to_numpy().astype(int), np.arange(len(df_item)))
        # )

        # func_each = lambda x: item_index_dict[x]
        # func_each = np.vectorize(func_each)
        # all_item_ind_array = func_each(to_go_item_array)
        
        series_tags_items = df_item[tag_label]
        
        target_df.loc[:,"diversity"] = get_diversiy(series_tags_items, to_go_item_array)



        # plot the distplot of seq_df["rating"]
        # sns.displot(target_df["rating"], kde=False, rug=False)
        # plt.savefig(os.path.join(FIGPATH, "rating_distplot.png"))
        # plt.close()

        # get the numpy format of cumulative distribution of seq_df["rating"], make sure the maximum is 1
        rating_cumsum = np.cumsum(
            target_df["rating"].value_counts().sort_index().to_numpy()
        ) / len(target_df)
        print(rating_cumsum)

        like_threshold = 5
        target_df.loc[:,"serendipity_5"] = get_serendipity(
            series_tags_items, to_go_item_array, to_go_rating_array, like_threshold
        )

        like_threshold = 4
        target_df.loc[:,"serendipity_4"] = get_serendipity(
            series_tags_items, to_go_item_array, to_go_rating_array, like_threshold
        )

        like_threshold = 3
        target_df.loc[:,"serendipity_3"] = get_serendipity(
            series_tags_items, to_go_item_array, to_go_rating_array, like_threshold
        )

        target_df.loc[:,"sum_rating"] = to_go_rating_array.sum(axis=1)

        # save seq_df
        

        return target_df
    
    
    





# def get_user_item_feat(user_feat_path, item_feat_path, tag_label="tags"):
#     df_user = pd.read_csv(
#         user_feat_path,
#         sep="\t",
#         header=0,
#         dtype={0: int, 1: int, 2: str, 3: int, 4: str},
#     )
#     # remove the last 6 tokens in each column name of df_user
#     df_user.columns = df_user.columns.str[:-6]
#     df_user.set_index("user_id", inplace=True)


#     df_item = pd.read_csv(
#         item_feat_path,
#         sep="\t",
#         header=0,
#         names=["item_id", "movie_title", "release_year", "genre"],
#         dtype={0: int, 1: str, 2: int, 3: str},
#     )
#     df_item.set_index("item_id", inplace=True)
    

#     # split df_item["genre"] into a list of genres
#     df_item["genre"] = df_item["genre"].apply(lambda x: x.split())
#     df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))

#     lbe_item_feat = LabelEncoder()
#     a_list_lists = df_item["genre"].to_list()
#     # chain a list of lists in a list
#     a_list = list(chain(*a_list_lists))
#     lbe_item_feat.fit(a_list)
#     df_item[tag_label] = df_item["genre"].apply(lambda x: lbe_item_feat.transform(x))

    



#     return df_user, df_item


# target_df, to_go_seq_dict



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
