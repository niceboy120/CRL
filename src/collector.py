import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# from buffer import ReplayBuffer
import sys
import logzero

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

    def compile(self, test_seq_length, target_pareto_index=None):

        self.test_seq_length = test_seq_length
        if self.test_seq_length > self.buffer_length:
            self.buffer_length = self.test_seq_length

        save_size = len(self.test_dataset.session_start_id_list) * self.buffer_length
        num_buffer = len(self.test_dataset.session_start_id_list)

        self.buffer = VectorReplayBuffer(save_size, num_buffer)

        if target_pareto_index is None:
            target_pareto_index = np.random.randint(len(self.env.pareto_front))

        self.target_pareto_index = target_pareto_index
        self.target_pareto = self.env.pareto_front[target_pareto_index]
        logzero.logger.info(f"pareto front: index: {target_pareto_index}, front: [{self.target_pareto}]")

        indices = np.array(self.test_dataset.session_start_id_list)

        x_batch = np.zeros((len(indices), 1, self.test_dataset.x_numpy.shape[-1]))
        seq_batch = np.zeros((len(indices), 1, *self.test_dataset.seq_numpy.shape[-2:]))
        y_batch = np.zeros((len(indices), 1, self.test_dataset.y_numpy.shape[-1]))
        len_data_batch = np.zeros(len(indices))

        # for idx_k, idx in tqdm(enumerate(indices), total=len(indices), desc="compiling test dataset"):
        for idx_k, idx in enumerate(indices):
            (x, seq, y, len_data) = self.test_dataset.__getitem__(idx)
            x[:len_data, self.test_dataset.user_index + 1:] = self.target_pareto  # Todo: note user_index=1 is the start index of target!
            x_batch[idx_k] = x[:len_data, :]
            seq_batch[idx_k] = seq[:len_data, :, :]
            y_batch[idx_k] = y[:len_data, :]
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

        for round in tqdm(range(self.test_seq_length), total=self.test_seq_length, desc="collecting test dataset"):
            # batch = self.buffer[self.buffer.last_index]
            batch, indices = self.buffer.sample(0)

            (x_batch, seq_batch, y_batch, len_data_batch) = batch.x_batch, batch.seq_batch, batch.y_batch, batch.len_data_batch

            x_batch = x_batch.reshape(self.buffer.buffer_num, -1, x_batch.shape[-1])
            seq_batch = seq_batch.reshape(self.buffer.buffer_num, -1, *seq_batch.shape[-2:])
            y_batch = y_batch.reshape(self.buffer.buffer_num, -1, y_batch.shape[-1])
            len_data_batch = len_data_batch.reshape(self.buffer.buffer_num, -1)[:, -1]

            # x = x[:, :len_data, :]

            assert all(len_data_batch == len_data_batch[0])
            # len_all = len_data_batch[0].astype(int)
            x_batch_tensor = torch.from_numpy(x_batch).float()
            seq_batch_tensor = torch.from_numpy(seq_batch).float()
            y_batch_tensor = torch.from_numpy(y_batch).int()
            len_data_batch_tensor = torch.from_numpy(len_data_batch).long()

            dataset = torch.utils.data.TensorDataset(x_batch_tensor, seq_batch_tensor, y_batch_tensor,
                                                     len_data_batch_tensor)
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
            y_pred[y_pred == 0] = np.random.randint(1, self.test_dataset.num_items, size=(y_pred == 0).sum()) # todo: for movielens-1m

            # Obtain reward:
            user_id = x_batch[:, round, self.test_dataset.user_index].astype(int)
            rewards = self.mat[user_id, y_pred].reshape(-1, 1)


            df_item_new = self.test_dataset.df_item.loc[y_pred].reset_index(drop=False)
            item_new = df_item_new[[col.name for col in self.test_dataset.seq_columns[:-1]]].to_numpy() # TODO: note that the last index (-1) indicates the rating column!
            item_reward = np.concatenate([item_new, rewards], axis=-1) # Todo: note that the last index (-1) indicates the rating column!

            # Update buffer:
            x_batch_new = x_batch[:, -1:, :].copy()

            seq_batch_new = seq_batch[:, -1:, :, :].copy()
            seq_batch_new[:, :, :, :-1] = seq_batch_new[:, :, :, 1:]
            seq_batch_new[:, -1, :, -1] = item_reward

            y_batch_new = y_batch[:, -1:, :].copy()
            y_batch_new[:, -1, :] = self.test_dataset.y_numpy[
                np.array(self.test_dataset.session_start_id_list) + round + 1]

            len_data_batch_new = len_data_batch + 1
            len_data_batch_new[len_data_batch_new > self.test_dataset.max_seq_length] = self.test_dataset.max_seq_length

            batch = Batch(x_batch=x_batch_new, seq_batch=seq_batch_new, y_batch=y_batch_new,
                          len_data_batch=len_data_batch_new)

            ptrs = self.buffer.add(batch)

            # round += 1

        res = self.env.step(self.buffer, self.test_seq_length)

        return res


        # self.test_dataset.seq_numpy[60400, 0]
