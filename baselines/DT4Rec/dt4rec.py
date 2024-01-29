# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from d2l import torch as d2l

# from timm.models.layers import trunc_normal_
from torch.nn.init import trunc_normal_

import sys

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from ctrl import CausalSelfAttention

sys.path.extend(["./src"])

from inputs import build_input_features, create_embedding_matrix, compute_input_dim, input_from_feature_columns

import torch
import numpy as np
from torch.utils.data import Dataset


class Autodis(nn.Module):
    def __init__(self, config, bucket_number):
        super().__init__()
        self.bucket = nn.Sequential(nn.Linear(1, config.n_embd))
        self.ret_emb_score = nn.Sequential(nn.Linear(1, bucket_number, bias=False), nn.LeakyReLU())
        self.res = nn.Linear(bucket_number, bucket_number, bias=False)
        self.temp = nn.Sequential(
            nn.Linear(1, bucket_number, bias=False),
            nn.LeakyReLU(),
            nn.Linear(bucket_number, bucket_number, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x, layer_past=None):
        bucket_value = torch.arange(0, 700, 7).to(x.device).reshape(100,1).type(torch.float32)
        Meta_emb = self.bucket(bucket_value)
        t = self.temp(x)
        x = self.ret_emb_score(x)
        # x.shape [batch_size, timestep, bucket_value]
        # t.shape [bucket_value]
        x = x + self.res(x)
        max_value,_ = torch.max(x, dim=2, keepdim=True)
        x = torch.exp(x - max_value)
        soft_sum = torch.sum(x, dim=2).unsqueeze(2)
        x = x / soft_sum
        x = torch.einsum('nck,km->ncm', [x, Meta_emb]) # torch.matmul(x, Meta_emb)
        return x


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.N_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlens * 3, config.max_seqlens * 3))
                             .view(1, 1, config.max_seqlens * 3, config.max_seqlens * 3))
        self.N_head = config.N_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.N_head, C // self.N_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.N_head, C // self.N_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.N_head, C // self.N_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True)

    def forward(self, X):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        # X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        # packed_input = pack_padded_sequence(X, len_hist_i, batch_first=True, enforce_sorted=False)
        packed_output, state = self.rnn(X)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return packed_output, state



def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class DT4Rec(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # D, iD = config.global_D, config.local_D
        # p1, p2 = config.patch_size
        # c, h, w = config.img_size
        # patch_count = h*w//p1//p2
        # max_seqlens = config.max_seqlens

        # input embedding stem
        self.drop = nn.Dropout(config.embd_pdrop)

        # self.blocks = nn.ModuleList([SABlock(config) for _ in range(config.n_layer)])

        self.local_pos_drop = nn.Dropout(config.pos_drop)
        self.global_pos_drop = nn.Dropout(config.pos_drop)
        self.device = config.device

        self.max_seqlens = config.max_seqlens
        self.ret_emb = Autodis(self.config, bucket_number=100)

        # self.tok_emb = nn.Embedding(config.num_item, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlens * 3 + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlens * 3, config.n_embd))
        self.state_encoder = Seq2SeqEncoder(config.num_item, config.n_embd, config.n_embd, 2, 0.2)
        self.action_encoder = Seq2SeqEncoder(config.num_item, config.n_embd, config.n_embd, 2, 0.2)

        # self.decoder = Seq2SeqDecoder(config.num_item, config.n_embd, config.n_embd, 2, 0.2)

        self.target_emb = nn.Embedding(self.config.num_item, self.config.n_embd)
        self.decoder = nn.Sequential(
            nn.Linear(self.config.n_embd * 2, self.config.n_embd),
            nn.ReLU(),
            nn.Linear(self.config.n_embd, self.config.n_embd),
        )
        # self.dense = nn.Linear(self.config.n_embd, self.config.num_item)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_head = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_item)

        print("number of parameters: %d" % sum(p.numel() for p in self.parameters()))

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        for i in (param_dict.keys() - union_params):
            no_decay.add(str(i))
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_loss(self, pred, target):
        if 'continuous' in self.config.action_type:
            return F.mse_loss(pred, target, reduction='none')
        else:
            return F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1).long(), reduction='none')

    def forward(self, x, reward, seq, targets, len_data, len_hist):

        # actions should be already padded by dataloader

        rtgs = reward.sum(2).unsqueeze(-1)

        states = seq[:, :, self.config.item_col_id]
        state_embeddings = torch.zeros([states.shape[0], states.shape[1], self.config.n_embd], device=seq.device)
        for i in range(states.shape[1]):
            states_seq = states[:, i, :].type(torch.long).squeeze(1)
            # len_hist_i = len_hist[:, i]
            packed_output, state = self.state_encoder(states_seq)
            # context = state.permute(1, 0, 2)
            state_embeddings[:, i, :] = state[-1]



        if targets is None:
            actions = states
        else:
            targets = targets.type(torch.long)
            actions = torch.zeros([states.size(0), states.size(1), self.max_seqlens], device=states.device)
            for i in range(states.size(0)):
                for j in range(states.size(1)):
                    if self.max_seqlens == len_hist[i, j]:
                        actions[i, j, :self.max_seqlens-1] = states[i, j, 1:]
                        actions[i, j, self.max_seqlens-1] = targets[i, j]
                    else:
                        actions[i, j, :len_hist[i, j]] = states[i, j, :len_hist[i, j]]
                        actions[i, j,  len_hist[i, j]] = targets[i, j]

            # actions = torch.concatenate([states, targets], dim=-1)
        action_embeddings = torch.zeros([actions.shape[0], actions.shape[1] - (1 if targets is None else 0), self.config.n_embd], device=actions.device)
        state_allstep = []

        for i in (range(actions.shape[1] - 1) if targets is None else range(actions.shape[1])):
            action_seq = actions[:, (i + 1) if targets is None else i, :].type(torch.long).squeeze(1)
            # len_hist_i = len_hist[:, i]
            output, state = self.action_encoder(action_seq)
            # context = state.permute(1, 0, 2)
            action_embeddings[:, i, :] = state[-1]
            state_allstep.append(output)

        rtg_neg = torch.zeros([rtgs.shape[0] * 8, rtgs.shape[1], rtgs.shape[2]], device=seq.device)
        for i in range(8):
            for j in range(rtgs.shape[0]):
                rtg_neg[i * rtgs.shape[0] + j, :-1, 0] = rtgs[j, 1:, 0] + i
                rtg_neg[i * rtgs.shape[0] + j, -1, 0] = rtgs[j, -1, 0]

        rtg_embeddings = self.ret_emb(rtgs)
        token_embeddings = torch.zeros(
            (rtg_embeddings.shape[0], rtg_embeddings.shape[1] * 3 - (1 if targets is None else 0), self.config.n_embd),
            dtype=torch.float32, device=rtg_embeddings.device)

        token_embeddings[:, ::3, :] = rtg_embeddings
        token_embeddings[:, 1::3, :] = state_embeddings
        token_embeddings[:, 2::3, :] = action_embeddings
        # token_embeddings[:, 2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]

        batch_size = token_embeddings.shape[0]
        # all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :].repeat_interleave(batch_size, dim=0)
        token_embeddings = token_embeddings.to(reward.device)

        if targets is None:
            x = self.drop(token_embeddings + position_embeddings)
            logits = self.blocks(x)
            target_logits = logits[:, -1, :].squeeze(1)  # only keep predictions from state_embeddings

            y = self.head(self.ln_head(target_logits))

            return y, None, None

        token_neg_embeddings = torch.repeat_interleave(token_embeddings, 8, 0)
        rtg_neg_embeddings = self.ret_emb(rtg_neg.type(torch.float32))
        token_neg_embeddings[:, ::3, :] = rtg_neg_embeddings


        token_all = torch.cat((token_embeddings, token_neg_embeddings), 0)
        position_all = torch.repeat_interleave(position_embeddings, 9, 0)

        x = self.drop(token_all + position_all)
        logits = self.blocks(x)

        target_logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings


        # loss_func = MaskedSoftmaxCELoss()
        loss_func = nn.CrossEntropyLoss()
        loss = []

        for i in range(actions.shape[1]):
            logits_new = target_logits[:, i, :].squeeze(1)
            targets_seq = targets[:, i, :].squeeze(1)
            # neg_seq = actions_neg[:, i, :].type(torch.long).squeeze(1)
            # pos_seq = actions[:, i, :].type(torch.long).squeeze(1)
            # bos = torch.tensor([0] * targets_seq.shape[0]).reshape(-1, 1).to(reward.device)
            # dec_input = torch.cat([bos, targets_seq], 1)

            logits_new_pos = logits_new[:actions.shape[0]]
            target_emb_pos = self.target_emb(targets[:, i].squeeze())
            context = torch.concatenate([target_emb_pos, logits_new_pos], dim=-1)
            Y_emb = self.decoder(context)
            Y_pred = self.head(self.ln_head(logits_new_pos))


            logits_new_neg = logits_new[actions.shape[0]:]
            target_emb_neg = target_emb_pos.repeat_interleave(8, 0)
            context_neg = torch.concatenate([target_emb_neg, logits_new_neg], dim=-1)
            Y_emb_neg = self.decoder(context_neg)

            pos_seq_emb = torch.gather(state_allstep[i], dim=1, index= (len_data - 1).view(len(len_data), 1, 1).expand(len(len_data), 1, state_allstep[i].shape[-1])).squeeze()

            pos_score = bleu_emb(Y_emb, pos_seq_emb)
            neg_score = bleu_emb(Y_emb.repeat_interleave(8, 0), Y_emb_neg)

            loss_step = loss_func(Y_pred, targets_seq)

            loss_all = loss_step - 0.1 * (pos_score - neg_score)

            loss.append(loss_all)

        loss_mean = sum(loss) / len(loss)

        return None, None, loss_mean

def bleu_emb(pred_seq, label_seq):  #@save
    """计算BLEU"""
    # retA=0

    score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))
    res = score_step.mean()

    return res

# ------------------------------------------------------------------------


class CTRLConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    action_type = "discrete"

    def __init__(self, num_item, **kwargs):
        self.num_item = num_item
        for k, v in kwargs.items():
            setattr(self, k, v)
        # assert self.img_size is not None and self.patch_size is not None
        # assert self.D % self.N_head == 0
        # C, H, W = self.img_size
        # pH, pW = self.patch_size


if __name__ == "__main__":
    pass

