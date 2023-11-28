

from environments.ML_1M.ml1m_data import get_seq_data_and_rewards
# for ml-1m


columns = ['user_id', 'item_id', 'rating', 'timestamp']
seq_df = get_seq_data_and_rewards(columns, max_item_list_len=10)


from environments.KuaiRand_1K.kuairand1k import get_seq_df_rts
columns=[
            "user_id",
            "item_id",
            "time_ms",
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "play_time_ms",
            "duration_ms",
            "profile_stay_time",
            "comment_stay_time",
            "is_profile_enter",
            "is_rand",
        ]
seq_df = get_seq_data_and_rewards(columns, max_item_list_len=10)