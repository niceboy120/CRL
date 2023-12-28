import os
import pickle
import pandas as pd

from .utils import get_seq_target_and_indices, get_statistics


class BaseData:

    def __init__(self, *args, **kwargs):
        self.DATAPATH = os.path.join(self.ENVPATH, "data")
        self.FIGPATH = os.path.join(self.ENVPATH, "figs")
        self.RESULTPATH = os.path.join(self.ENVPATH, "data_processed")
        for path in [self.FIGPATH, self.RESULTPATH]:
            if not os.path.exists(path):
                os.mkdir(path)

    # @staticmethod
    def get_and_save_seq_data(self, df_data, df_user, df_item, x_columns, reward_columns, seq_columns, max_item_list_len, len_reward_to_go, reload=False):
        filepath_seq_rewards = os.path.join(self.RESULTPATH, "df_seq_rewards.csv")
        filepath_hist_seq_dict = os.path.join(self.RESULTPATH, "hist_seq_dict.pickle")
        filepath_to_go_seq_dict = os.path.join(self.RESULTPATH, "to_go_seq_dict.pickle")

        if (
                not reload
                and os.path.exists(filepath_seq_rewards)
                and os.path.exists(filepath_hist_seq_dict)
                and os.path.exists(filepath_to_go_seq_dict)
        ):
            print("Data has already been processed! Loading...")
            df_seq_rewards = pd.read_csv(filepath_seq_rewards)
            hist_seq_dict = pickle.load(open(filepath_hist_seq_dict, "rb"))
            to_go_seq_dict = pickle.load(open(filepath_to_go_seq_dict, "rb"))
            print("Data loaded!")

        else:
            # df, df_user, df_item, list_feat = self.get_data()
            df_user_part = df_user.reset_index()[[col.name for col in x_columns if col.name in df_user.reset_index()]].set_index("user_id")
            df_item_part = df_item.reset_index()[[col.name for col in seq_columns if col.name in df_item.reset_index()]].set_index("item_id")
            df_data = df_data.join(df_user_part, on="user_id", how="left")
            df_data = df_data.join(df_item_part, on="item_id", how="left")
            df_seq_rewards, hist_seq_dict, to_go_seq_dict = self.get_seq_data_and_rewards(
                df_data, df_user, df_item, seq_columns, max_item_list_len, len_reward_to_go)

            # df_seq_rewards = pd.read_csv(os.path.join(self.RESULTPATH, "seq_df_rts.csv"))
            df_seq_rewards.to_csv(filepath_seq_rewards, index=False)
            pickle.dump(hist_seq_dict, open(filepath_hist_seq_dict, "wb"))
            pickle.dump(to_go_seq_dict, open(filepath_to_go_seq_dict, "wb"))

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict

    # @staticmethod
    def get_seq_data_and_rewards(self, df, df_user, df_item, seq_columns, max_item_list_len, len_reward_to_go, tag_label="tags"):
        import warnings
        from numba.core.errors import NumbaPendingDeprecationWarning
        warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        # Step 1: get target dataframe and sequences and to go sequences
        target_df, hist_seq_dict, to_go_seq_dict = get_seq_target_and_indices(df, seq_columns, max_item_list_len, len_reward_to_go)

        # Step 2: get the reward statistics.
        df_seq_rewards = get_statistics(df, df_user, df_item, target_df, to_go_seq_dict,
                                        serendipity_threshold=self.serendipity_threshold, tag_label=tag_label)

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict





