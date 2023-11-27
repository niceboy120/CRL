import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# from buffer import ReplayBuffer
import sys
sys.path.extend(["./src/data_structure"])

from batch import Batch
from buffer_manager import VectorReplayBuffer


class Collector:
    def __init__(self, env, policy, test_dataset, mat, buffer_length=20) -> None:
        self.env = env
        self.policy = policy
        self.test_dataset = test_dataset
        self.mat = mat
        self.buffer_length = buffer_length

    def compile(self, test_seq_length, target_pareto_index=None, test_seq_step=1):
        
        self.test_seq_length = test_seq_length
        if self.test_seq_length > self.buffer_length:
            self.buffer_length = self.test_seq_length
        self.test_seq_step = test_seq_step

        save_size = len(self.test_dataset.session_start_id_list) * self.buffer_length
        num_buffer = len(self.test_dataset.session_start_id_list)

        self.buffer = VectorReplayBuffer(save_size, num_buffer)

        if target_pareto_index is None:
            target_pareto_index = np.random.randint(len(self.env.pareto_front))

        self.target_pareto_index = target_pareto_index
        self.target_pareto = self.env.pareto_front[target_pareto_index]
        
        indices = np.array(self.test_dataset.session_start_id_list)

        x_batch = np.zeros((len(indices), self.test_dataset.max_seq_length, self.test_dataset.x_numpy.shape[-1]))
        seq_batch = np.zeros((len(indices), self.test_dataset.max_seq_length, *self.test_dataset.seq_numpy.shape[-2:]))
        y_batch = np.zeros((len(indices), self.test_dataset.max_seq_length, self.test_dataset.y_numpy.shape[-1]))
        len_data_batch = np.zeros(len(indices))
        
        # for idx_k, idx in tqdm(enumerate(indices), total=len(indices), desc="compiling test dataset"):
        for idx_k, idx in enumerate(indices):
            (x, seq, y, len_data) = self.test_dataset.__getitem__(idx)
            x[:len_data, self.test_dataset.user_index:] = self.target_pareto # Todo: note user_index=1 is the start index of target!
            x_batch[idx_k] = x
            seq_batch[idx_k] = seq
            y_batch[idx_k] = y
            len_data_batch[idx_k] = len_data

        batch = Batch(x_batch=x_batch, seq_batch=seq_batch, y_batch=y_batch, len_data_batch=len_data_batch)
        # batch = Batch(obs=obs, obs_next=np_ui_pair, act=items, is_start=is_starts,
        #               policy={}, info={}, rew=rewards, rew_prev=rew_prevs, terminated=terminateds, truncated=truncateds)
        
        ptrs = self.buffer.add(batch)
        
        
    def collect(self, batch_size=1000, num_workers=4):

        # self.loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
        #                          batch_size=batch_size, num_workers=num_workers)
        self.policy.eval()

        # self.buffer[0]
        # batch, index = self.buffer.sample(0)
        batch = self.buffer[self.buffer.last_index]
        (x_batch, seq_batch, y_batch, len_data_batch) = batch.x_batch, batch.seq_batch, batch.y_batch, batch.len_data_batch
        
        # x = x[:, :len_data, :]
        
        round = 0
        
        
        # assert all(len_data_batch == len_data_batch[0])
        len_all = len_data_batch[0].astype(int)
        x_batch_tensor = torch.from_numpy(x_batch[:, :len_all, :]).float()
        seq_batch_tensor = torch.from_numpy(seq_batch[:, :len_all, :, :]).float()
        y_batch_tensor = torch.from_numpy(y_batch[:, :len_all, :]).int()
        # len_data_batch = torch.from_numpy(len_data_batch[:len_all]).long()
        len_data_batch_tensor = torch.from_numpy(len_data_batch).long()
        
        dataset = torch.utils.data.TensorDataset(x_batch_tensor, seq_batch_tensor, y_batch_tensor, len_data_batch_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=num_workers)

        act_logit_list = []
        with torch.no_grad():
            for (x, seq, y, len_data) in dataloader:
                x = x.to(self.policy.device)
                seq = seq.to(self.policy.device)
                y = y.to(self.policy.device)
                len_data = len_data.to(self.policy.device)
                
                act_logit, atts, loss = self.policy(x, seq, targets=None, len_data=len_data)
                act_logit_list.append(act_logit)
        
        y_logits = torch.cat(act_logit_list, dim=0)
        
        # Greedy action:
        y_pred = torch.argmax(y_logits, dim=-1).cpu().numpy()
        
        # Obtain reward:
        user_id = x_batch[:, round, self.test_dataset.user_index].astype(int)
        rewards = self.mat[user_id, y_pred]
        

        
        
        batch = Batch(x_batch=x_batch, seq_batch=seq_batch, y_batch=y_batch, len_data_batch=len_data_batch)

        
        


        
        
        # self.test_dataset.seq_numpy[60400, 0]
        
        
        
        
        
        
        
        
        
        


    