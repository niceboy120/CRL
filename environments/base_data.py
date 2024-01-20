import os
import pickle
import pandas as pd

from .utils import get_seq_target_and_indices, get_statistics


class BaseData:

    def __init__(self, *args, **kwargs):
        self.item_padding_id = None
        self.user_padding_id = None
        self.DATAPATH = os.path.join(self.ENVPATH, "data")
        self.FIGPATH = os.path.join(self.ENVPATH, "figs")
        self.RESULTPATH = os.path.join(self.ENVPATH, "data_processed")
        for path in [self.FIGPATH, self.RESULTPATH]:
            if not os.path.exists(path):
                os.mkdir(path)

    # @staticmethod
    # NOTE: add augment rate
    def get_and_save_seq_data(self, df_data, df_user, df_item, x_columns, reward_columns, seq_columns, max_item_list_len, len_reward_to_go, reload=False, augment_rate=0):
        postfix = '' if augment_rate == 0 else f'_+{augment_rate}'
        filepath_seq_rewards = os.path.join(self.RESULTPATH, f"df_seq_rewards{postfix}.csv")
        filepath_hist_seq_dict = os.path.join(self.RESULTPATH, f"hist_seq_dict{postfix}.pickle")
        filepath_to_go_seq_dict = os.path.join(self.RESULTPATH, f"to_go_seq_dict{postfix}.pickle")

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

    @staticmethod
    def data_augment(df_data, time_field_name, augment_rate=1):
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
            sampled[time_field_name] = 0
            sampled['date'] = pd.to_datetime(sampled[time_field_name], unit='s')
            sampled['day'] = sampled['date'].dt.date
            auxiliary_data.append(sampled)
        auxiliary_df = pd.concat(auxiliary_data)
        # add to original data
        augmented_df = pd.concat([df_data, auxiliary_df])
        augmented_df.sort_values(by=["user_id", time_field_name], ascending=True, inplace=True)
        return augmented_df

