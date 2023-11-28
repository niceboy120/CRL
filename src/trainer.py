import math
import os

import logzero
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

writer = None


class TrainerConfig:
    # optimization parameters, will be overried by given actual parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights

    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9

    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, collector, train_dataset, test_dataset, config):
        self.model = model
        self.collector = collector
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = torch.device(f'cuda:{config.args.cuda_id}')
        #     # self.model = torch.nn.DataParallel(self.model).to(self.device)
        #     self.model = self.model.to(self.device)
        self.device = config.args.device

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            dataset = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            data_len = len(loader)
            pbar = tqdm(enumerate(loader), total=data_len) if is_train else enumerate(loader)

            for it, (x, seq, y, len_data) in pbar:
                x = x.to(self.device)  # states
                seq = seq.to(self.device)  # action
                y = y.to(self.device)  # reward
                len_data = len_data.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):

                    act_logit, atts, loss = model(x, seq, targets=y, len_data=len_data)
                    # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

                    if is_train:
                        for p in model.parameters():
                            p.grad = None
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        if writer is not None:
                            writer.add_scalar('training_loss', loss.item(), epoch_num * data_len + it)

                if is_train:
                    if config.lr_decay:
                        self.tokens += (seq >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs + 1):
            # res = self.collector.collect()  # for debug
            # eval_return, eval_std = self.get_returns(0) # for debug
            if epoch > 0:
                run_epoch('train', epoch_num=epoch)

            res = self.collector.collect()
            logzero.logger.info(f"Epoch: [{epoch}], Info: [{res}]")
            if writer is not None:
                for key, value in res.items():
                    writer.add_scalar(key, value, epoch)

    def test(self):

        self.model.train(False)

    def get_returns(self, ret):
        self.model.train(False)

        dataset = self.test_dataset
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=1000,  # self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        data_len = len(loader)

        pbar = tqdm(enumerate(loader), total=data_len)  # if is_train else enumerate(loader)

        for it, (x, seq, y, len_data) in pbar:
            x = x.to(self.device)  # states
            seq = seq.to(self.device)  # action
            y = y.to(self.device)  # reward
            len_data = len_data.to(self.device)

            # forward the model
            with torch.set_grad_enabled(False):
                # self.env.reset(x, y, len_data)

                act_logit, atts, loss = self.model(x, seq, targets=y, len_data=len_data)

        # envargs = EnvArgs(self.config.game.lower(), self.config.seed, self.config.img_size[-2:])
        # env = Env(envargs)
        # env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            # state = env.reset()
            all_states = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            rewards = []
            actions = []

            # padding
            actions += [self.config.num_items]
            rewards += [0]
            # sampled_action = sample(self.model.module, all_states, sample=True,
            #                                   actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
            #                                   rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0) if 'rtg' not in self.config.model_type else torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0))
            sampled_action = sample(self.model, all_states, sample=True,
                                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                    rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0) if 'rtg' not in self.config.model_type else torch.tensor(rtgs,
                                                                                                                 dtype=torch.float32).to(
                                        self.device).unsqueeze(1).unsqueeze(0))
            j = 0
            while True:
                if done:
                    state, reward_sum, done, prev_attn = env.reset(), 0, False, None
                    all_states = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                    actions = []
                    rewards = []
                    rtgs = [ret]

                    # padding
                    actions += [self.config.num_items]
                    rewards += [0]

                # take a step
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                rewards += [reward]
                reward_sum += reward
                state = state.unsqueeze(0).to(self.device)
                rtgs += [rtgs[-1] - reward]

                # trunk trajectory
                all_states = torch.cat([all_states, state.unsqueeze(0)], dim=1)
                if all_states.size(1) > self.config.maxT:
                    all_states = all_states[:, -self.config.maxT:]
                    actions = actions[-self.config.maxT:]
                    rewards = rewards[-self.config.maxT:]
                    rtgs = rtgs[-self.config.maxT:]
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                sampled_action = sample(self.model, all_states, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(
                                            1).unsqueeze(0) if 'rtg' not in self.config.model_type else torch.tensor(
                                            rtgs, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0))

        env.close()
        eval_return = sum(T_rewards) / 10.
        eval_std = np.std(T_rewards)
        print("eval return: %d, eval std: %f" % (eval_return, eval_std))

        self.model.train(True)
        return eval_return, eval_std


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, sample=False, top_k=None, actions=None, rewards=None):
    model.eval()
    logits, _, _ = model(x, actions, rewards=rewards)

    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)

    x = ix

    return x
