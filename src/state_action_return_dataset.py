import torch
import numpy as np
from torch.utils.data import Dataset


class StateActionReturnDataset(Dataset):

    def __init__(self, df_user, df_item, x_columns, y_column, seq_columns, max_seq_length, user_index=0):
        self.df_user = df_user
        self.df_item = df_item
        self.x_columns = x_columns
        self.y_column = y_column
        self.seq_columns = seq_columns
        self.max_seq_length = max_seq_length
        self.user_index = user_index
        
        self.num_users = x_columns[0].vocabulary_size
        self.num_items = y_column.vocabulary_size
    
    def compile(self, df_seq_rewards, hist_seq_dict, to_go_seq_dict):
        
        self.x_names = []

        x_list = [df_seq_rewards[column.name].to_numpy().reshape(-1, 1) for column in self.x_columns]
        self.x_numpy = np.concatenate(x_list, axis=-1)
        user_ids = self.x_numpy[:,0]
        self.session_start_id_list = [0] + list(np.diff(user_ids).nonzero()[0] + 1)
        
        seq_list = [hist_seq_dict[column.name + "_list"] for column in self.seq_columns]
        seq_numpy_temp = np.stack(seq_list) # (num_fields, num_samples, seq_length)
        self.seq_numpy = np.transpose(seq_numpy_temp, (1, 0, 2)) # (num_samples, num_fields, seq_length)

        self.y_numpy = df_seq_rewards[self.y_column.name].to_numpy().reshape(-1, 1)

        # The last id corresponds to the padding id!
        self.x_numpy = np.concatenate([self.x_numpy, np.zeros_like(self.x_numpy[0:1])], axis=0)
        self.seq_numpy = np.concatenate([self.seq_numpy, np.zeros_like(self.seq_numpy[0:1])], axis=0)
        self.y_numpy = np.concatenate([self.y_numpy, np.zeros_like(self.y_numpy[0:1])], axis=0)
        self.padding_idx = len(self.y_numpy) - 1

        self.x_numpy = self.x_numpy.astype(np.float32)
        self.seq_numpy = self.seq_numpy.astype(np.float32)
        self.y_numpy = self.y_numpy.astype(np.int64)
        
        self.to_go_seq_dict = to_go_seq_dict # for evaluation todo

        # self.x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        # self.seq_tensor = torch.tensor(seq_numpy, dtype=torch.float32)


    def get_torch_dataset(self):
        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
        #                                          torch.from_numpy(self.seq_numpy),
        #                                          torch.from_numpy(self.y_numpy))
        dataset = torch.utils.data.TensorDataset(torch.Tensor(self.x_numpy),
                                                 torch.Tensor(self.seq_numpy),
                                                 torch.Tensor(self.y_numpy))
        return dataset

    def __getitem__(self, idx):
        
        start_idx = idx - self.max_seq_length + 1 
        indices = np.arange(start_idx, idx + 1)
    
        id_left, id_right = np.searchsorted(self.session_start_id_list, [start_idx, idx], "right")
        if id_right != id_left:
            session_start_id = self.session_start_id_list[id_right - 1]
            # print("start", session_start_id)
            indices[:session_start_id - start_idx] = self.padding_idx
            indices = np.roll(indices, start_idx - session_start_id)
            len_data = self.max_seq_length + start_idx - session_start_id
        else:
            len_data = self.max_seq_length

        return self.x_numpy[indices], self.seq_numpy[indices], self.y_numpy[indices], len_data

    def __len__(self):
        return len(self.y_numpy) - 1 # minus 1 because of padding.

