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

    def run_epoch(self, model, config, optimizer, epoch, split, epoch_num=0):
        is_train = split == 'train'
        model.train(is_train)
        dataset = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        losses = []
        data_len = len(loader)
        pbar = tqdm(enumerate(loader), total=data_len) if is_train else enumerate(loader)

        for it, (x, reward, seq, y, len_data, len_hist) in pbar:
            x = x.to(self.device)  # states
            reward = reward.to(self.device)  # desired reward info
            seq = seq.to(self.device)  # sequence data
            y = y.to(self.device)  # next target
            len_data = len_data.to(self.device)
            len_hist = len_hist.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):

                act_logit, atts, loss = model(x, reward, seq, targets=y, len_data=len_data, len_hist=len_hist)
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

            pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs + 1):
            if epoch > 0:
                self.run_epoch(model, config, optimizer, epoch, split='train', epoch_num=epoch)
            res = self.collector.collect()
            logzero.logger.info(f"Epoch: [{epoch}/{config.max_epochs}], Info: [{res}]")
            if writer is not None:
                for key, value in res.items():
                    writer.add_scalar(key, value, epoch)

    def test(self):
        self.model.train(False)



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
