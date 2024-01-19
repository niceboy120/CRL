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


class Collector_Baseline_SL:
    def __init__(self, env, policy, test_dataset, mat, test_seq_length, device="cpu", buffer_length=20) -> None:
        self.env = env
        self.policy = policy
        self.test_dataset = test_dataset
        self.mat = mat
        self.buffer_length = buffer_length
        self.test_seq_length = test_seq_length
        self.device = device
        if self.test_seq_length > self.buffer_length:
            self.buffer_length = self.test_seq_length

        if env.item_padding_id is not None:
            self.mat = np.delete(self.mat, env.item_padding_id, axis=1)
        if env.user_padding_id is not None:
            self.mat = np.delete(self.mat, env.user_padding_id, axis=0)


    def compile(self, target_pareto_index=None):

        save_size = len(self.test_dataset.session_start_id_list) * self.buffer_length
        num_buffer = len(self.test_dataset.session_start_id_list)

        # self.buffer = VectorReplayBuffer(save_size, num_buffer)

        # if target_pareto_index is None:
        #     target_pareto_index = np.random.randint(len(self.env.pareto_front))

        # self.target_pareto_index = target_pareto_index
        # self.target_pareto = self.env.pareto_front[target_pareto_index]
        # logzero.logger.info(f"pareto front: index: {target_pareto_index}, front: [{self.target_pareto}]")

        # indices = np.array(self.test_dataset.session_start_id_list)
        # cate_batch = np.zeros((len(indices), self.test_dataset.categorical_numpy.shape[-1]))
        # reward_batch = np.zeros((len(indices), self.test_dataset.reward_numpy.shape[-1]))
        # seq_batch = np.zeros((len(indices), 1, *self.test_dataset.seq_numpy.shape[-2:]))

        # x_batch = np.zeros((len(indices), 1, self.test_dataset.x_numpy.shape[-1]))
        # reward_batch = np.zeros((len(indices), 1, self.test_dataset.reward_numpy.shape[-1]))
        # seq_batch = np.zeros((len(indices), 1, *self.test_dataset.seq_numpy.shape[-2:]))
        # y_batch = np.zeros((len(indices), 1, self.test_dataset.y_numpy.shape[-1]))
        # len_data_batch = np.zeros(len(indices))

        ## Put one batch in the buffer, and set the reward to the target pareto front!
        # for idx_k, idx in enumerate(indices):
        #     (x, rewards, seq, y, len_data) = self.test_dataset.__getitem__(idx)
        #     rewards[:len_data] = self.target_pareto
        #     x_batch[idx_k] = x[:len_data, :]
        #     reward_batch[idx_k] = rewards[:len_data, :]
        #     seq_batch[idx_k] = seq[:len_data, :, :]
        #     y_batch[idx_k] = y[:len_data, :]
        #     len_data_batch[idx_k] = len_data
        #
        # batch = Batch(x_batch=x_batch, reward_batch=reward_batch, seq_batch=seq_batch, y_batch=y_batch, len_data_batch=len_data_batch)
        # ptrs = self.buffer.add(batch)

    def collect(self, user_batch_size=2, num_workers=4):

        self.policy.eval()
        self.compile()

        # for round in tqdm(range(self.test_seq_length), total=self.test_seq_length, desc="collecting test dataset"):
        all_hist_item = []
        all_feedback = []

        # user_idx_buffer = np.zeros([user_batch_size], np.int64)
        # user_feat_buffer = np.zeros([user_batch_size, self.test_dataset.user_numpy.shape[1]])
        # now_id = 0

        for user_idx, user_feat in tqdm(enumerate(self.test_dataset.user_numpy), desc="collecting user feedback for all users", total=len(self.test_dataset.user_numpy)):

            # if now_id < user_batch_size:
            #     user_idx_buffer[now_id] = user_idx
            #     user_feat_buffer[now_id] = user_feat
            #     now_id += 1
            #     if now_id < user_batch_size:
            #         if user_idx < len(self.test_dataset.user_numpy) - 1:
            #             continue
            #         else:
            #             user_idx_buffer = user_idx_buffer[:now_id]
            #             user_feat_buffer = user_feat_buffer[:now_id]
            # now_id = 0

            item_feat = self.test_dataset.item_numpy

            # cate_feat = np.array([np.concatenate([user[self.test_dataset.cate_user_idx], item[self.test_dataset.cate_item_idx]], axis=-1) for user in user_feat_buffer for item in item_feat])
            # num_feat = np.array([np.concatenate([user[self.test_dataset.num_user_idx], item[self.test_dataset.num_item_idx]], axis=-1) for user in user_feat_buffer for item in item_feat])
            cate_feat = np.array([np.concatenate([user_feat[self.test_dataset.cate_user_idx], item[self.test_dataset.cate_item_idx]], axis=-1) for item in item_feat])
            num_feat = np.array([np.concatenate([user_feat[self.test_dataset.num_user_idx], item[self.test_dataset.num_item_idx]], axis=-1) for item in item_feat])

            cate_feat_torch = torch.from_numpy(cate_feat).long().to(self.device)
            num_feat_torch = torch.from_numpy(num_feat).float().to(self.device)
            if num_feat_torch.shape[-1] == 0:
                num_feat_torch = torch.zeros([len(num_feat_torch), 1], dtype=torch.float).to(self.device)


            y = self.policy(cate_feat_torch, num_feat_torch)
            y_sum = torch.vstack(y).sum(0)
            # logit_mask = get_logit_mask(y_sum, hist_item_indices)
            # y_logits_masked = y_logits * logit_mask
            # y_logits_masked = torch.where(y_sum == 0, torch.full_like(y_sum, float('-inf')), y_sum)
            # y_logits_masked = y_sum.reshape(user_batch_size, -1)
            y_logits_masked = y_sum


            # Greedy action:
            # y_pred = torch.argmax(y_logits_masked, dim=-1).cpu().numpy()
            y_pred = torch.topk(y_logits_masked, self.test_seq_length, dim=-1).indices.cpu().numpy()

            # Obtain reward:
            # feedback = np.array([self.mat[user_id, y_pred[k]] for k, user_id in enumerate(user_idx)])
            feedback = self.mat[user_idx, y_pred]

            all_hist_item.append(y_pred)
            all_feedback.append(feedback)

        all_hist_item_numpy = np.array(all_hist_item)
        all_feedback_numpy = np.array(all_feedback)
        res = self.env.evaluate(all_hist_item_numpy, all_feedback_numpy)

        return res

def get_logit_mask(y_logits, hist_item_indices):
    logit_mask = torch.ones_like(y_logits, dtype=torch.bool).to(device=y_logits.device)
    logit_mask[hist_item_indices] = 0

    return logit_mask