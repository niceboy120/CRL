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
    def __init__(self, env, policy, test_dataset, mat, test_seq_length, buffer_length=20) -> None:
        self.env = env
        self.policy = policy
        self.test_dataset = test_dataset
        self.mat = mat
        self.buffer_length = buffer_length
        self.test_seq_length = test_seq_length
        if self.test_seq_length > self.buffer_length:
            self.buffer_length = self.test_seq_length

    def compile(self, target_pareto_index=None):

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
        reward_batch = np.zeros((len(indices), 1, self.test_dataset.reward_numpy.shape[-1]))
        seq_batch = np.zeros((len(indices), 1, *self.test_dataset.seq_numpy.shape[-2:]))
        y_batch = np.zeros((len(indices), 1, self.test_dataset.y_numpy.shape[-1]))
        len_data_batch = np.zeros(len(indices))

        ## Put one batch in the buffer, and set the reward to the target pareto front!
        for idx_k, idx in enumerate(indices):
            (x, rewards, seq, y, len_data) = self.test_dataset.__getitem__(idx)
            rewards[:len_data] = self.target_pareto
            x_batch[idx_k] = x[:len_data, :]
            reward_batch[idx_k] = rewards[:len_data, :]
            seq_batch[idx_k] = seq[:len_data, :, :]
            y_batch[idx_k] = y[:len_data, :]
            len_data_batch[idx_k] = len_data

        batch = Batch(x_batch=x_batch, reward_batch=reward_batch, seq_batch=seq_batch, y_batch=y_batch, len_data_batch=len_data_batch)
        ptrs = self.buffer.add(batch)

    def collect(self, batch_size=1000, num_workers=4):

        self.policy.eval()
        self.compile()

        for round in tqdm(range(self.test_seq_length), total=self.test_seq_length, desc="collecting test dataset"):
            batch, indices = self.buffer.sample(0)

            (x_batch, reward_batch, seq_batch, y_batch, len_data_batch) = batch.x_batch, batch.reward_batch, batch.seq_batch, batch.y_batch, batch.len_data_batch

            x_batch = x_batch.reshape(self.buffer.buffer_num, -1, x_batch.shape[-1])
            reward_batch = reward_batch.reshape(self.buffer.buffer_num, -1, reward_batch.shape[-1])
            seq_batch = seq_batch.reshape(self.buffer.buffer_num, -1, *seq_batch.shape[-2:])
            y_batch = y_batch.reshape(self.buffer.buffer_num, -1, y_batch.shape[-1])
            len_data_batch = len_data_batch.reshape(self.buffer.buffer_num, -1)[:, -1]

            # x = x[:, :len_data, :]

            assert all(len_data_batch == len_data_batch[0])
            # len_all = len_data_batch[0].astype(int)
            x_batch_tensor = torch.from_numpy(x_batch).float()
            reward_batch_tensor = torch.from_numpy(reward_batch).float()
            seq_batch_tensor = torch.from_numpy(seq_batch).float()
            y_batch_tensor = torch.from_numpy(y_batch).int()
            len_data_batch_tensor = torch.from_numpy(len_data_batch).long()

            dataset = torch.utils.data.TensorDataset(
                x_batch_tensor, reward_batch_tensor, seq_batch_tensor, y_batch_tensor, len_data_batch_tensor)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)

            act_logit_list = []
            with torch.no_grad():
                for (x, reward, seq, y, len_data) in dataloader:
                    x = x.to(self.policy.device)
                    reward = reward.to(self.policy.device)
                    seq = seq.to(self.policy.device)
                    y = y.to(self.policy.device)
                    len_data = len_data.to(self.policy.device)

                    act_logit, atts, loss = self.policy(x, reward, seq, targets=None, len_data=len_data)
                    act_logit_list.append(act_logit)

            y_logits = torch.cat(act_logit_list, dim=0)

            logit_mask = get_logit_mask(y_logits, self.buffer, indices, self.env.item_col, self.env.item_padding_id)
            # y_logits_masked = y_logits * logit_mask
            y_logits_masked = torch.where(logit_mask == 0, torch.full_like(y_logits, float('-inf')), y_logits)

            # Greedy action:
            y_pred = torch.argmax(y_logits_masked, dim=-1).cpu().numpy()


            # Obtain reward:
            user_id = x_batch[:, round, self.test_dataset.user_index].astype(int)
            feedback = self.mat[user_id, y_pred].reshape(-1, 1)

            df_item_new = self.test_dataset.df_item.loc[y_pred].reset_index(drop=False)
            item_new = df_item_new[[col.name for col in self.test_dataset.seq_columns[:-1]]].to_numpy() # TODO: note that the last index (-1) indicates the user feedback!
            item_feedback = np.concatenate([item_new, feedback], axis=-1) # Todo: note that the last index (-1) indicates the user feedback!

            # Update buffer:
            x_batch_new = x_batch[:, -1:, :].copy()
            reward_batch_new = reward_batch[:, -1:, :].copy()

            seq_batch_new = seq_batch[:, -1:, :, :].copy()
            seq_batch_new[:, :, :, :-1] = seq_batch_new[:, :, :, 1:]
            seq_batch_new[:, -1, :, -1] = item_feedback

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

def get_rec_ids(buffer, indices, item_col):
    if len(buffer) == 0:
        rec_id = None
    else:
        rec_ids = buffer.seq_batch[indices][:, :,item_col].reshape(buffer.buffer_num, -1)
    return rec_ids

def get_logit_mask(y_logits, buffer, indices, item_col, item_padding_id):
    logit_mask = torch.ones_like(y_logits, dtype=torch.bool).to(device=y_logits.device)
    rec_ids = get_rec_ids(buffer, indices, item_col)
    rec_ids_torch = torch.LongTensor(rec_ids).to(device=y_logits.device)
    logit_mask = logit_mask.scatter(1, rec_ids_torch, 0)
    if item_padding_id is not None:
        logit_mask[:, item_padding_id] = 0  # todo: for movielens-1m

    return logit_mask



