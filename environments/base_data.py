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
    def get_and_save_seq_data(self, df_data, df_user, df_item, x_columns, reward_columns, seq_columns, max_item_list_len, len_reward_to_go, 
                              reload=False, time_field_name='timestamp', augment_type='mat', augment_rate=0, augment_strategies=['random']):
        postfix = '' if augment_rate == 0 else f'_{augment_type}+{augment_rate}_' + '_'.join(augment_strategies)
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
                df_data, df_user, df_item, seq_columns, max_item_list_len, len_reward_to_go,
                time_field_name, augment_type, augment_rate, augment_strategies)

            # df_seq_rewards = pd.read_csv(os.path.join(self.RESULTPATH, "seq_df_rts.csv"))
            df_seq_rewards.to_csv(filepath_seq_rewards, index=False)
            pickle.dump(hist_seq_dict, open(filepath_hist_seq_dict, "wb"))
            pickle.dump(to_go_seq_dict, open(filepath_to_go_seq_dict, "wb"))

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict

    # @staticmethod
    def get_seq_data_and_rewards(self, df, df_user, df_item, seq_columns, max_item_list_len, len_reward_to_go, 
                                 time_field_name, augment_type, augment_rate, augment_strategies, tag_label="tags"):
        import warnings
        from numba.core.errors import NumbaPendingDeprecationWarning
        warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        # Step 1: get target dataframe and sequences and to go sequences
        target_df, hist_seq_dict, to_go_seq_dict = get_seq_target_and_indices(df, seq_columns, max_item_list_len, len_reward_to_go)

        # Step 1.1 (optional): insert augmented data to extracted sequence dataset
        if augment_type == 'seq':
            print("Augment Sequence Data")
            target_df, hist_seq_dict, to_go_seq_dict = self.augment_sequence(df, target_df, df_item, hist_seq_dict, to_go_seq_dict, 
                                                                             seq_columns, time_field_name, augment_rate, augment_strategies)

        # Step 2: get the reward statistics.
        df_seq_rewards = get_statistics(df, df_user, df_item, target_df, to_go_seq_dict,
                                        serendipity_threshold=self.serendipity_threshold, tag_label=tag_label)

        return df_seq_rewards, hist_seq_dict, to_go_seq_dict

    # @staticmethod
    def augment_matrix(self, df:pd.DataFrame, df_item:pd.DataFrame, time_field_name:str, augment_rate:float=1, strategies:list[str]=['random']):
        from tqdm import tqdm
        import numpy as np
        if augment_rate <= 0:
            return df
        # load augmented data
        postfix = f'_+{augment_rate}_' + '_'.join(strategies)
        filepath_augment_df = os.path.join(self.RESULTPATH, f"df_augment{postfix}.csv")
        if os.path.exists(filepath_augment_df):
            print('Augment data already generated! Loading...')
            return pd.read_pickle(filepath_augment_df)
        all_user_id = df['user_id'].unique()
        insert_index = 0
        for uid in tqdm(all_user_id, desc="Iterating for augmentation"):
            user_interactions = df[df['user_id'] == uid]
            user_item_num = len(user_interactions)
            augment_item_num = int(user_item_num * augment_rate)
            auxiliary_data = []
            for strategy in strategies:
                if strategy == 'random':
                    # randomly select new interactions
                    sampled = user_interactions.sample(n=augment_item_num, replace=True, random_state=42)
                elif strategy == 'rating':
                    # maximize rating: generate interactions consisting of only max-rating items
                    max_interactions = user_interactions[user_interactions['rating'] == user_interactions['rating'].max()]
                    sampled = max_interactions.sample(n=augment_item_num, replace=True, random_state=42)
                elif strategy == 'diversity':
                    # randomly sample an item and greedily add items ?
                    user_interactions = user_interactions.join(df_item, on="item_id", how="left")
                    user_interactions['tags'] = user_interactions['tags'].apply(lambda x: set(x))  # to set
                    sampled = []
                    cur_tags, candidate_interactions = set(), user_interactions
                    while len(sampled) < augment_item_num:
                        if len(candidate_interactions) == 0:
                            # re-sample interaction, reset tags
                            cur_inter = user_interactions.iloc[np.random.randint(user_item_num)]
                            cur_tags = cur_inter['tags']
                        else:
                            # select from candidate interaction, update tags
                            cur_inter = candidate_interactions.iloc[np.random.randint(len(candidate_interactions))]
                            cur_tags = cur_tags | cur_inter['tags']
                        sampled.append(cur_inter)
                        overlapped = user_interactions['tags'].apply(lambda x: len(cur_tags & x))
                        candidate_interactions = user_interactions[overlapped == 0]
                    # to dataframe
                    sampled = pd.DataFrame(sampled)[[col for col in df.columns]]
                else:
                    raise NotImplementedError(f"Unknown augmentation strtegy: {strategy}")
                # set sampled's timestamp to 0 (so that it will always appears before real interactions when sorting)
                sampled[time_field_name] = 0
                sampled['date'] = pd.to_datetime(sampled[time_field_name], unit='s')
                sampled['day'] = sampled['date'].dt.date
                auxiliary_data.append(sampled)
            # concat auxiliary data at the beginning of each user
            auxiliary_df = pd.concat(auxiliary_data)
            assert df.iloc[insert_index]['user_id'] == uid
            df = pd.concat([df.iloc[:insert_index], auxiliary_df, df.iloc[insert_index:]]).reset_index(drop=True)
            insert_index += user_item_num + len(auxiliary_df)
        # save augment df
        df.to_pickle(filepath_augment_df)
        return df

    def augment_sequence(self, df_data:pd.DataFrame, target_df:pd.DataFrame, df_item:pd.DataFrame,
                         hist_seq_dict:dict, to_go_seq_dict:dict, seq_columns:list,
                         time_field_name:str, augment_rate:float=0.1, strategies:list[str]=['random']):
        from tqdm import tqdm
        import numpy as np

        if augment_rate == 0:
            return target_df, hist_seq_dict, to_go_seq_dict
        # augment sequential data
        hist_len, to_go_len = hist_seq_dict['item_id_list'].shape[-1], to_go_seq_dict['item_id_list'].shape[-1]
        augment_data = {}
        augment_idxs = np.arange(len(target_df))[np.random.uniform(size=len(target_df)) < augment_rate]
        orig_len_per_uid = target_df.groupby('user_id').size()
        for idx, uid in tqdm(zip(augment_idxs, target_df.iloc[augment_idxs]['user_id']), 
                             total=len(augment_idxs), desc='generate augmentation data'):
            # get all user's available items
            user_interactions = df_data[df_data['user_id'] == uid]
            user_items = user_interactions['item_id']
            # history recommended items
            hist_item_list = hist_seq_dict['item_id_list'][idx].tolist()
            recommended = set(hist_item_list)
            available_mask = user_items.apply(lambda x: x not in recommended)  # not recommended
            # check available items
            if len(user_items[available_mask].unique()) < (to_go_len * 2 - 1):
                continue
            if not uid in augment_data:
                augment_data[uid] = {
                    'orig_idx': idx,
                    'aug_len': 0,
                    'hist_dict': [],
                    'to_go_dict': [],
                    'target': [],
                }
            # generate specific sequence for augmentation, then a random sequence for padding
            for strategy in strategies:
                if strategy == 'random':
                    # random pick from available items
                    selected_inters = user_interactions[available_mask].sample(n=to_go_len*2-1, replace=False)
                else:
                    if strategy == 'rating':
                        # pick items with the highest rating
                        optimal_inters = user_interactions[available_mask].nlargest(n=to_go_len, columns='rating', keep='all')
                        optimal_inters = optimal_inters.sample(n=to_go_len)  # so it's random for each state
                    elif strategy == 'diversity':
                        # greedily select items to (NOTE:approximately) maximize togo's diversity
                        available_interactions = user_interactions[available_mask].join(df_item[['tags']], on='item_id', how='left')
                        available_interactions['tags'] = available_interactions['tags'].apply(lambda x: set(x))
                        if False:
                            # TODO: if we take history items into account, then the initial tags set should be history tags
                            # here is an example
                            hist_mask = user_items.apply(lambda x: x in hist_item_list[-5:])  # recent
                            hist_interactions = user_interactions[hist_mask].join(df_item, on='item_id', how='left')
                            hist_tags = set(hist_interactions['tags'].explode().to_list())
                        optimal_inters = []
                        cur_tags, candidate_interactions = set(), available_interactions
                        for _ in range(to_go_len):
                            if len(candidate_interactions) == 0:
                                # now that all item tags has been used, just randomly select one (no repeat)
                                available_items = available_interactions['item_id']
                                optimal_items = [inter['item_id'] for inter in optimal_inters]
                                available_interactions = available_interactions[available_items.apply(lambda x: x not in optimal_items)]
                                cur_inter = available_interactions.iloc[np.random.randint(len(available_interactions))]
                                cur_tags = cur_inter['tags']
                            else:
                                # select from candidate interaction, update tags
                                cur_inter = candidate_interactions.iloc[np.random.randint(len(candidate_interactions))]
                                cur_tags = cur_tags | cur_inter['tags']
                            optimal_inters.append(cur_inter)
                            overlapped = available_interactions['tags'].apply(lambda x: len(cur_tags & x))
                            candidate_interactions = available_interactions[overlapped == 0]
                        optimal_inters = pd.DataFrame(optimal_inters)[[col for col in df_data.columns]]
                    # at last we randomly select items to pad
                    optimal_items = optimal_inters['item_id'].to_list()
                    rest_mask = available_mask & user_items.apply(lambda x: x not in optimal_items)  # remove previously selected
                    selected_inters = pd.concat([optimal_inters, user_interactions[rest_mask].sample(n=to_go_len-1, replace=False)])
                assert len(selected_inters) == 2 * to_go_len - 1
                # set selected inters' time to 0 (actually this seems useless now)
                selected_inters[time_field_name] = 0
                selected_inters['date'] = pd.to_datetime(selected_inters[time_field_name], unit='s')
                selected_inters['day'] = selected_inters['date'].dt.date
                # convert selected interactions into sequential data (hist, target & to_go)
                hist_dict = {f'{col.name}_list':np.zeros([to_go_len, hist_len]) for col in seq_columns}
                to_go_dict = {f'{col.name}_list':np.zeros([to_go_len, to_go_len]) for col in seq_columns}
                valid_len = hist_item_list.index(0) if 0 in hist_item_list else hist_len
                cur_hist_dict = {f'{col.name}_list':hist_seq_dict[f'{col.name}_list'][idx][:valid_len].tolist()
                                 for col in seq_columns}
                last_item_data = {f'{col.name}_list':[] for col in seq_columns}
                for cur_idx in range(to_go_len):
                    # TODO: check the item's order inside seq data
                    for col in seq_columns:
                        key_name = f'{col.name}_list'
                        # update cur_hist data (a manually updated window) 
                        cur_hist_dict[key_name] = (cur_hist_dict[key_name] + last_item_data[key_name])[-hist_len:]
                        # set seq values
                        to_go_dict[key_name][cur_idx] = selected_inters.iloc[cur_idx:cur_idx+to_go_len][col.name].to_numpy().flatten()
                        hist_dict[key_name][cur_idx][:len(cur_hist_dict[key_name])] = cur_hist_dict[key_name]
                        # update last item data
                        last_item_data[key_name] = [selected_inters.iloc[cur_idx][col.name]]
                augment_data[uid]['hist_dict'].append(hist_dict)
                augment_data[uid]['to_go_dict'].append(to_go_dict)
                augment_data[uid]['target'].append(selected_inters.iloc[:to_go_len])
                augment_data[uid]['aug_len'] += to_go_len
        # insert generated data into original hist_seq, target_df & to_go_seq
        aug_data_len = sum([info['aug_len'] for info in augment_data.values()])
        new_hist_dict = {key: np.empty([aug_data_len + len(target_df), hist_len]) for key in hist_seq_dict}
        new_to_go_dict = {key: np.empty([aug_data_len + len(target_df), to_go_len]) for key in hist_seq_dict}
        orig_start_idx, aug_start_idx = 0, 0
        candidate_users = target_df['user_id'].unique()
        for uid in tqdm(candidate_users, desc='insert augmentaion data'):
            assert target_df.iloc[aug_start_idx]['user_id'] == uid
            orig_len = orig_len_per_uid[uid]
            if uid in augment_data:
                augmentation = augment_data[uid]
                aug_len = augmentation['aug_len']
                # calculate idx
                orig_end_idx = orig_start_idx + orig_len
                raw_start_idx = aug_start_idx + aug_len
                raw_end_idx = raw_start_idx + orig_len
                # insert to seq dict
                for col in seq_columns:
                    key = f'{col.name}_list'
                    # augment data & original data
                    tmp_hist_data = np.concatenate([tmp_dict[key] for tmp_dict in augmentation['hist_dict']])
                    tmp_to_go_data = np.concatenate([tmp_dict[key] for tmp_dict in augmentation['to_go_dict']])
                    orig_hist_data = hist_seq_dict[key][orig_start_idx: orig_end_idx]
                    orig_to_go_data = to_go_seq_dict[key][orig_start_idx: orig_end_idx]
                    # save into new np array
                    new_hist_dict[key][aug_start_idx:raw_start_idx] = tmp_hist_data
                    new_to_go_dict[key][aug_start_idx:raw_start_idx] = tmp_to_go_data
                    new_hist_dict[key][raw_start_idx:raw_end_idx] = orig_hist_data
                    new_to_go_dict[key][raw_start_idx:raw_end_idx] = orig_to_go_data
                # insert to target_df
                cur_target = pd.concat(augmentation['target'])
                target_df = pd.concat([target_df[:aug_start_idx], cur_target, target_df[aug_start_idx:]])
            else:
                orig_end_idx = orig_start_idx + orig_len
                raw_start_idx = aug_start_idx
                raw_end_idx = raw_start_idx + orig_len
                # insert to seq dict
                for col in seq_columns:
                    key = f'{col.name}_list'
                    # original data
                    orig_hist_data = hist_seq_dict[key][orig_start_idx: orig_end_idx]
                    orig_to_go_data = to_go_seq_dict[key][orig_start_idx: orig_end_idx]
                    # save to new data
                    new_hist_dict[key][raw_start_idx:raw_end_idx] = orig_hist_data
                    new_to_go_dict[key][raw_start_idx:raw_end_idx] = orig_to_go_data
            # increase index
            orig_start_idx = orig_end_idx
            aug_start_idx = raw_end_idx
        target_df.reset_index(drop=True, inplace=True)
        return target_df, new_hist_dict, new_to_go_dict