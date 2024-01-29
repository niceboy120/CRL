import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.extend(["./src/data_structure"])




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


    def collect(self, user_batch_size=2, num_workers=4):

        self.policy.eval()
        # self.compile()

        all_hist_item = []
        all_feedback = []
        for user_idx, user_feat in tqdm(enumerate(self.test_dataset.user_numpy), desc="collecting user feedback for all users", total=len(self.test_dataset.user_numpy)):
            item_feat = self.test_dataset.item_numpy

            cate_feat = np.array([np.concatenate([user_feat[self.test_dataset.cate_user_idx], item[self.test_dataset.cate_item_idx]], axis=-1) for item in item_feat])
            num_feat = np.array([np.concatenate([user_feat[self.test_dataset.num_user_idx], item[self.test_dataset.num_item_idx]], axis=-1) for item in item_feat])

            cate_feat_torch = torch.from_numpy(cate_feat).long().to(self.device)
            num_feat_torch = torch.from_numpy(num_feat).float().to(self.device)
            if num_feat_torch.shape[-1] == 0:
                num_feat_torch = torch.zeros([len(num_feat_torch), 1], dtype=torch.float).to(self.device)

            y = self.policy(cate_feat_torch, num_feat_torch)

            y_sum = torch.vstack(y).sum(0)
            y_logits_masked = y_sum


            # Greedy action:
            # y_pred = torch.argmax(y_logits_masked, dim=-1).cpu().numpy()
            y_pred = torch.topk(y_logits_masked, self.test_seq_length, dim=-1).indices.cpu().numpy()

            # Obtain reward:
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