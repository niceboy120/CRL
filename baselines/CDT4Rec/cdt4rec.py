# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.layers import trunc_normal_
from torch.nn.init import trunc_normal_

import sys

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.extend(["./src"])

from inputs import build_input_features, create_embedding_matrix, compute_input_dim, input_from_feature_columns


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return F.gelu(input) 

class CausalSelfAttention(nn.Module):
    def __init__(self, config, N_head=8, D=128, T=30):
        super().__init__()
        assert D % N_head == 0
        self.config = config
        

        self.N_head = N_head
        
        self.key = nn.Linear(D, D)
        self.query = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resd_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(D, D)
    def forward(self, x, mask=None):
        # x: B * N * D
        B, N, D = x.size()
        
        q = self.query(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        k = self.key(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        v = self.value(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        
        A = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            A = A.masked_fill(mask[:,:,:N,:N] == 0, float('-inf'))
        A = F.softmax(A, dim=-1)
        A_drop = self.attn_drop(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.resd_drop(self.proj(y))
        return y, A


    
class _SABlock(nn.Module):
    def __init__(self, config, N_head, D):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(config, N_head=N_head, D=D)
        self.mlp = nn.Sequential(
                nn.Linear(D, 4*D),
                GELU(),
                nn.Linear(4*D, D),
                nn.Dropout(config.resid_pdrop)
            )

        
    def forward(self, x, mask=None):
        y, att = self.attn(self.ln1(x), mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att
    
    
class SABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # p1, p2 = config.patch_size
        # c, h, w = config.img_size
        # patch_count = h*w//p1//p2

        self.local_block = _SABlock(config, config.local_N_head, config.local_D)
        self.global_block = _SABlock(config, config.N_head, config.global_D)

        if 'fusion' in config.model_name:
            self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlens*2, config.max_seqlens*2))
                                     .view(1, 1, config.max_seqlens*2, config.max_seqlens*2))
        else:
            self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlens*2, config.max_seqlens*2))
                                     .view(1, 1, config.max_seqlens*2, config.max_seqlens*2))
            for i in range(0, config.max_seqlens*2, 2):
                self.mask[0, 0, i, i-1] = 1.

        self.local_norm = nn.LayerNorm(config.local_D)
        if 'stack' not in config.model_name:
            self.local_global_proj = nn.Sequential(
                    nn.Linear(config.local_num_feat * config.local_D, config.global_D), # +1 for action token
                    nn.LayerNorm(config.global_D)
            )
        
        
    def forward(self, local_tokens, global_tokens, temporal_emb=None):

        B, T, P, d = local_tokens.size()
        local_tokens, local_att = self.local_block(local_tokens.reshape(-1, P, d))
        local_tokens = local_tokens.reshape(B, T, P, d)
        lt_tmp = self.local_norm(local_tokens.reshape(-1, d)).reshape(B*T, P*d)
        lt_tmp = self.local_global_proj(lt_tmp).reshape(B, T, -1)
        if 'fusion' in self.config.model_name or 'xconv' in self.config.model_name:
            global_tokens += lt_tmp
            if ('xconv' in self.config.model_name) and (temporal_emb is not None):
                global_tokens += temporal_emb
            global_tokens, global_att = self.global_block(global_tokens, self.mask)
            return local_tokens, global_tokens, local_att, global_att
        else:
            if temporal_emb is not None:
                lt_tmp += temporal_emb
            global_tokens = torch.stack((lt_tmp, global_tokens), dim=2).view(B, -1, self.config.global_D)
            global_tokens, global_att = self.global_block(global_tokens, self.mask)
            return local_tokens, global_tokens[:, 1::2], local_att, global_att
                


    
class PatchEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # D, iD = config.global_D, config.local_D
        # p1, p2 = config.patch_size
        # c, h, w = config.img_size

        self.max_seqlens = config.max_seqlens

        self.sequence_embedding = nn.GRU(
            input_size=compute_input_dim(config.seq_columns),
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True
        )


        self.user_encoder = nn.Sequential(
            nn.Linear(compute_input_dim(config.x_columns), 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_size)
        )


        self.feature_embedding = create_embedding_matrix(config.x_columns, init_std=0.0001, linear=False, sparse=False, device='cpu')
        self.reward_embedding = create_embedding_matrix(config.reward_columns, init_std=0.0001, linear=False, sparse=False, device='cpu')
        self.seq_embedding = create_embedding_matrix(config.seq_columns, init_std=0.0001, linear=False, sparse=False, device='cpu')
        self.action_embedding = nn.Embedding(config.seq_columns[config.col_item].vocabulary_size, config.seq_columns[config.col_item].embedding_dim, sparse=False, padding_idx=config.seq_columns[config.col_item].padding_idx)

        self.feature_index = build_input_features(config.x_columns)
        self.reward_index = build_input_features(config.reward_columns)
        self.seq_index = build_input_features(config.seq_columns)
        
        self.global_proj = nn.Sequential(nn.Linear(config.local_D, config.global_D), # +1 for action token
                                         nn.LayerNorm(config.global_D))

        self.temporal_emb = nn.Parameter(torch.zeros(1, config.max_seqlens, config.global_D))

    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'spatial_emb', 'temporal_emb'}
    
    def forward(self, x, reward, seq, len_data, len_hist):
        # B, T, C, H, W = x.size()
        # local_state_tokens = (self.sequence_embedding(x) + self.spatial_emb).reshape(B, T, -1, self.config.local_D)
        
        num_batch, len_data_with_padding, num_features = x.shape
        num_batch, len_data_with_padding, num_rewards = reward.shape
        num_batch, len_data_with_padding, num_seq_features, len_state = seq.shape

        action = seq[:, :, self.config.col_item, -1].type(torch.LongTensor).to(reward.device)

        x_values = input_from_feature_columns(x.reshape(-1, num_features), self.config.x_columns, self.feature_embedding, self.feature_index, support_dense=True, device=x.device)
        reward_values = input_from_feature_columns(reward.reshape(-1, num_rewards), self.config.reward_columns, self.reward_embedding, self.reward_index, support_dense=True, device=x.device)
        seq_values = input_from_feature_columns(seq.reshape(-1, num_seq_features, len_state), self.config.seq_columns, self.seq_embedding, self.seq_index, support_dense=True, device=x.device)
        # action_values = input_from_feature_columns(action.reshape(-1, 1), self.action_columns, self.action_embedding, self.action_index, support_dense=True, device=x.device)

        len_hist_reshape = len_hist.reshape(-1)

        mask = torch.arange(reward.size(1)).expand(len(len_data), reward.size(1)).to(len_data.device) < len_data.unsqueeze(1)
        mask = mask.view(-1)

        # from einops.layers.torch import Rearrange

        # x_tensor = combined_dnn_input(x_values, [])
        # seq_tensor = combined_dnn_input(seq_values, [])

        x_tensor = torch.cat(x_values, dim=1).reshape(num_batch, len_data_with_padding, num_features, -1)
        reward_tensor = torch.cat(reward_values, dim=1).reshape(num_batch, len_data_with_padding, num_rewards, -1)
        seq_tensor = torch.cat(seq_values, dim=-1).squeeze(1)
        action_tensor = self.action_embedding(action).unsqueeze(2)

        user_representation = self.user_encoder(x_tensor)


        seq_tensor_masked = seq_tensor[mask]
        len_hist_masked = len_hist_reshape[mask]

        packed_input = pack_padded_sequence(seq_tensor_masked, len_hist_masked.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hn = self.sequence_embedding(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # seq_tensor.detach().cpu().numpy()[:,-1,0]

        gru_output = torch.zeros([num_batch * len_data_with_padding, hn.shape[-1]]).to(reward.device)
        gru_output[mask] = hn[-1]
        # gru_output[mask] = output[:, -1]

        gru_output_recovered = gru_output.reshape(num_batch, len_data_with_padding, 1, -1)

        local_tokens = torch.cat([reward_tensor, user_representation, gru_output_recovered, action_tensor], dim=2)

        global_tokens = self.global_proj(gru_output_recovered.squeeze(2))

        return local_tokens, global_tokens, self.temporal_emb[:, :len_data_with_padding]



class CDT4Rec(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # D, iD = config.global_D, config.local_D
        # p1, p2 = config.patch_size
        # c, h, w = config.img_size
        # patch_count = h*w//p1//p2
        # max_seqlens = config.max_seqlens
        
        self.token_emb = PatchEmb(config)
        
        self.blocks = nn.ModuleList([SABlock(config) for _ in range(config.n_layer)])
        
        self.local_pos_drop = nn.Dropout(config.pos_drop)
        self.global_pos_drop = nn.Dropout(config.pos_drop)
        self.device = config.device

        if 'stack' in config.model_name:
            self.local_global_proj = nn.Sequential(
                    nn.Linear(config.local_num_feat * config.local_D, config.global_D), # +1 for action token
                    nn.LayerNorm(config.global_D)
            )
        self.ln_head = nn.LayerNorm(config.global_D)
        if 'continuous' in config.action_type:
            self.head = nn.Sequential(
                    *([nn.Linear(config.global_D, config.num_item)] + [nn.Tanh()])
                )
        else:
            self.head = nn.Linear(config.global_D, config.num_item)
        
        self.apply(self._init_weights)
        
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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                
                if "sequence_embedding" in pn: # only for sequence embedding
                    if "weight" in pn:
                        decay.add(fpn)
                    elif "bias" in pn:
                        no_decay.add(fpn)


        # no_decay.add('token_emb.spatial_emb')
        no_decay.add('token_emb.temporal_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

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

        local_tokens, global_state_tokens, temporal_emb = self.token_emb(x, reward, seq, len_data, len_hist)
        local_tokens = self.local_pos_drop(local_tokens)

        global_state_tokens = self.global_pos_drop(global_state_tokens)
        
        B, T, P, d = local_tokens.size()
        local_atts, global_atts = [], []

        for i, blk in enumerate(self.blocks):
            local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens, temporal_emb)
        
        # Create the mask using len_data
        mask = torch.arange(T).unsqueeze(0).to(len_data.device) >= len_data.unsqueeze(1)
        # Apply the mask to global_state_tokens
        global_state_tokens = global_state_tokens.masked_fill(mask.unsqueeze(-1), 0)
        
        y = self.head(self.ln_head(global_state_tokens))
        loss, loss_mean = None, None
        if targets is not None:
            loss = self.get_loss(y, targets)
            loss_mask = loss.masked_fill(mask.reshape(-1), 0)
            loss_mean = loss_mask.sum() / (~mask.reshape(-1)).sum() # get the masked average loss
            
        y_last = torch.gather(y, 1, len_data.long().view(-1, 1, 1).expand(-1, 1, y.size(2))-1).squeeze(1)
        return y_last, (local_atts, global_atts), loss_mean
        # return y[:, -1], (local_atts, global_atts), loss_mean
        
    
#------------------------------------------------------------------------
    
        
class CDT4RecConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    action_type = "discrete"

    def __init__(self, num_item, **kwargs):
        self.num_item = num_item
        for k,v in kwargs.items():
            setattr(self, k, v)
        # assert self.img_size is not None and self.patch_size is not None
        # assert self.D % self.N_head == 0
        # C, H, W = self.img_size
        # pH, pW = self.patch_size
        
        
if __name__ == "__main__":
    mconf = CDT4RecConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
                          N_head=8, D=192, local_N_head=4, local_D=64, model_name='star', max_timestep=100, n_layer=6, C=4, max_seqlens=30)

    model = CDT4Rec(mconf)
    model = model.cuda()
    dummy_states = torch.randn(3, 28, 4, 84, 84).cuda()
    dummy_actions = torch.randint(0, 4, (3, 28, 1), dtype=torch.long).cuda()
    output, atn, loss = model(dummy_states, dummy_actions, None)
    print (output.size(), output)

    dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
    dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
    output, atn, loss = model(dummy_states, dummy_actions, None)
    print (output.size(), output)

