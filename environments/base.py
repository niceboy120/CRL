import os
import pickle
import numpy as np

import pandas as pd
from tqdm import tqdm
from numba import njit


from .get_pareto_front import get_kde_density, get_pareto_front


class BaseEnv:
    def __init__(self, df_seq_rewards, bin=100, percentile=90, pareto_reload=False):
        
        user_features, item_features, reward_features = self.__class__.get_features()
        target_features = reward_features

        pareto_front_path = f"pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        raw_pareto_front_path = f"raw_pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        pareto_front_filepath = os.path.join(self.RESULTPATH, pareto_front_path)
        raw_pareto_front_filepath = os.path.join(self.RESULTPATH, raw_pareto_front_path)

        # if not reload and os.path.exists(pareto_front_filepath):
        #     print("pareto front has already been computed! Loading...")
        #     with open(pareto_front_filepath, "rb") as f:
        #         self.pareto_front = pickle.load(f)
        #     print("pareto front loaded!")
        #     return self.pareto_front
        if not pareto_reload and os.path.exists(raw_pareto_front_filepath): # use raw pareto front
            print("raw pareto front has already been computed! Loading...")
            with open(raw_pareto_front_filepath, "rb") as f:
                self.pareto_front = pickle.load(f)
            print("raw pareto front loaded!")
            # return self.raw_pareto_front
        else:
            density_flat, grid_flat = get_kde_density(df_seq_rewards, target_features, bin)
            self.pareto_front = get_pareto_front(density_flat, grid_flat, percentile)
            print(f"pareto front for bin:{bin} and percentile:{percentile} is ", self.pareto_front)
            # pickle dump pareto_front
            with open(pareto_front_filepath, "wb") as f:
                pickle.dump(self.pareto_front, f)
            
            # get raw pareto front
            key_statistics = df_seq_rewards.describe()[target_features]
            self.pareto_front = (key_statistics.loc["max"] - key_statistics.loc["min"]).to_numpy() * self.pareto_front + key_statistics.loc["min"].to_numpy()
            print(f"Raw pareto front for bin:{bin} and percentile:{percentile} is ", self.pareto_front)

            
            # pickle dump pareto_front
            with open(raw_pareto_front_filepath, "wb") as f:
                pickle.dump(self.pareto_front, f)

            # return self.raw_pareto_front

    def reset(self, ):
        pass

    def step(self,):
        pass
        
    


    @staticmethod
    def get_and_save_seq_data(EnvClass, RESULTPATH, seq_columns, max_item_list_len, len_reward_to_go, reload=False):
        filepath_seq_rewards = os.path.join(RESULTPATH, "df_seq_rewards.csv")
        filepath_hist_seq_dict = os.path.join(RESULTPATH, "hist_seq_dict.pickle")
        filepath_to_go_seq_dict = os.path.join(RESULTPATH, "to_go_seq_dict.pickle")

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
            df, df_user, df_item, list_feat = EnvClass.get_data()
            df = df.join(df_user, on="user_id", how="left")
            df = df.join(df_item, on="item_id", how="left")
            df_seq_rewards, hist_seq_dict, to_go_seq_dict = \
                BaseEnv.get_seq_data_and_rewards(EnvClass, df, df_user, df_item, seq_columns, max_item_list_len, len_reward_to_go)
            
            # df_seq_rewards = pd.read_csv(os.path.join(RESULTPATH, "seq_df_rts.csv"))
            df_seq_rewards.to_csv(filepath_seq_rewards, index=False)
            pickle.dump(hist_seq_dict, open(filepath_hist_seq_dict, "wb"))
            pickle.dump(to_go_seq_dict, open(filepath_to_go_seq_dict, "wb"))

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict

    @staticmethod
    def get_seq_data_and_rewards(
        EnvClass,
        df,
        df_user,
        df_item,
        seq_columns,
        max_item_list_len,
        len_reward_to_go,
        tag_label="tags",
    ):
        import warnings
        from numba.core.errors import NumbaPendingDeprecationWarning

        warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        # Step 1: get target dataframe and sequences and to go sequences
        target_df, hist_seq_dict, to_go_seq_dict = get_seq_target_and_indices(df, seq_columns, max_item_list_len, len_reward_to_go)

        # Step 2: get the reward statistics.
        df_seq_rewards = EnvClass.get_statistics(df, df_user, df_item, target_df, to_go_seq_dict, tag_label=tag_label)

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict



def get_seq_target_and_indices(df, seq_columns, max_item_list_len, len_reward_to_go):
    last_uid = None
    uid_list, hist_item_list_indices, to_go_item_list_indices, target_indices, item_list_length = [], [], [], [], []
    seq_start = 0
    for i, uid in enumerate(tqdm(df["user_id"].to_numpy(), desc="getting sequences' indices...", total=len(df))):
        if last_uid != uid:
            last_uid = uid
            seq_start = i
        else:
            if i - seq_start > max_item_list_len:
                seq_start += 1

            target_idx = i - len_reward_to_go

            uid_list.append(uid)
            hist_item_list_indices.append((seq_start, target_idx))
            to_go_item_list_indices.append((target_idx, i))
            target_indices.append(target_idx)
            item_list_length.append(i - seq_start)

    uid_list = np.array(uid_list)
    hist_item_list_indices = np.array(hist_item_list_indices)
    to_go_item_list_indices = np.array(to_go_item_list_indices)
    target_indices = np.array(target_indices)
    item_list_length = np.array(item_list_length, dtype=np.int64)

    # Only use the part of sequences that satisfied the minimum length!
    reserved_idx = item_list_length >= len_reward_to_go

    uid_list = uid_list[reserved_idx]
    hist_item_list_indices = hist_item_list_indices[reserved_idx]
    to_go_item_list_indices = to_go_item_list_indices[reserved_idx]
    target_indices = target_indices[reserved_idx]
    item_list_length = item_list_length[reserved_idx]

    new_length = len(hist_item_list_indices)
    target_df = df.iloc[target_indices]
    target_df.reset_index(drop=True, inplace=True)

    def get_seq_field_data():
        hist_seq_dict = dict()
        to_go_seq_dict = dict()

        hist_shape = (new_length, max_item_list_len - len_reward_to_go)
        to_go_shape = (new_length, len_reward_to_go)

        cols = seq_columns
        for field in cols:
            if field != "user_id":
                list_field = f"{field}_list"

                hist_seq_dict[list_field] = np.zeros(hist_shape)
                to_go_seq_dict[list_field] = np.zeros(to_go_shape)

                values = df[field].to_numpy()
                print(f"getting sequences field data: {field} ...")

                res_hist_seq_field = hist_seq_dict[list_field]
                res_to_go_seq_field = to_go_seq_dict[list_field]
                
                get_seq_field(
                    res_hist_seq_field,
                    res_to_go_seq_field,
                    len_reward_to_go,
                    values,
                    hist_item_list_indices,
                    to_go_item_list_indices,
                    item_list_length,
                )

        return hist_seq_dict, to_go_seq_dict

    hist_seq_dict, to_go_seq_dict = get_seq_field_data()

    return target_df, hist_seq_dict, to_go_seq_dict


@njit
def get_seq_field(
    res_hist_seq_field,
    res_to_go_seq_field,
    len_reward_to_go,
    values,
    hist_item_list_indices,
    to_go_item_list_indices,
    item_list_length,
):
    for i, (hist_slice, to_go_slice, length) in enumerate(
        zip(hist_item_list_indices, to_go_item_list_indices, item_list_length)
    ):
        res_hist_seq_field[i][: length - len_reward_to_go] = values[hist_slice[0]:hist_slice[1]]
        res_to_go_seq_field[i] = values[to_go_slice[0] : to_go_slice[1]]

    # return res_hist_seq_field
