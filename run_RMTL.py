import time
import traceback

import logzero
import torch
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
# pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
# from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import sys

sys.path.extend(["./src"])

from collector_SL import Collector_Baseline_SL
from data import get_DataClass, get_xy_columns, get_common_args, get_train_test_idx
from environments.SL_env import SLEnv
from inputs import SparseFeat, DenseFeat
from utils import prepare_dir_log
import argparse
sys.path.extend(["baselines/RMTL"])

from baselines.RMTL.train.utils import RlLossPolisher
from baselines.RMTL.train.run import one_iter_offline, sltrain
from baselines.RMTL.agents.DDPG_ESMM_BC import TD3_ESMMBCAgent
from baselines.RMTL.train.Arguments import Arguments



def get_SL_args():
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='rt',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', "rt", "rsc"])
    parser.add_argument('--dataset_path', default='baselines/RMTL/dataset/')
    parser.add_argument('--actor_model_name', type=str, default='esmm',
                        choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'esmm'])
    parser.add_argument('--model_name', type=str, default='RMTL')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--polish', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--feature_map_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--cuda', type=int, default=7)
    # parser.add_argument('--save_dir', default='saved_models/SL')

    # added arguments_for_CRL
    # ###########################################################################
    parser.add_argument("--env", type=str, default="ml-1m")
    # parser.add_argument("--env", type=str, default="Zhihu-1M")
    # parser.add_argument("--env", type=str, default="KuaiRand-Pure")
    parser.add_argument("--is_reload", dest="reload", action="store_true")
    parser.add_argument("--no_reload", dest="reload", action="store_false")
    parser.set_defaults(reload=False)

    parser.add_argument("--max_item_list_len", type=int, default=30)
    parser.add_argument("--len_reward_to_go", type=int, default=10)

    parser.add_argument('--load_message', type=str, default='SL')
    parser.add_argument('--message', type=str, default='run')
    # ###########################################################################

    args = parser.parse_known_args()[0]
    args = get_common_args(args)

    args.model_name = f"{args.model_name}_{args.actor_model_name}"
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)
    return args

class RL_Dataset(Dataset):
    def __init__(self, df_seq_rewards, df_user, df_item, user_columns, item_columns, reward_columns, user_index=0):
        self.df_seq_rewards = df_seq_rewards
        self.df_user = df_user
        self.df_item = df_item
        self.user_columns = user_columns
        self.item_columns = item_columns
        self.reward_columns = reward_columns
        self.user_index = user_index

        self.num_users = user_columns[0].vocabulary_size
        self.num_items = item_columns[0].vocabulary_size

        self.categorical_columns = [column for column in user_columns + item_columns if isinstance(column, SparseFeat)]
        self.numerical_columns = [column for column in user_columns + item_columns if isinstance(column, DenseFeat)]

        self.cate_user_idx = [i for i, col in enumerate(user_columns) if isinstance(col, SparseFeat)]
        self.cate_item_idx = [i for i, col in enumerate(item_columns) if isinstance(col, SparseFeat)]
        self.num_user_idx = [i for i, col in enumerate(user_columns) if isinstance(col, DenseFeat)]
        self.num_item_idx = [i for i, col in enumerate(item_columns) if isinstance(col, DenseFeat)]

        categorical_data_list = [df_seq_rewards[column.name].to_numpy().reshape(-1, 1) for column in self.categorical_columns]
        numerical_data_list = [df_seq_rewards[column.name].to_numpy().reshape(-1, 1) for column in self.numerical_columns]
        reward_data_list = [df_seq_rewards[column.name].to_numpy().reshape(-1, 1) for column in self.reward_columns]

        self.categorical_numpy = np.concatenate(categorical_data_list, axis=-1).astype(np.int64)
        self.numerical_numpy = np.concatenate(numerical_data_list, axis=-1).astype(np.float32) if len(numerical_data_list) else np.zeros([len(self.categorical_numpy), 1]).astype(np.float32)
        self.reward_numpy = np.concatenate(reward_data_list, axis=-1)

        self.session_numpy = df_seq_rewards["user_id"].to_numpy().astype(np.int64)

        user_ids = df_seq_rewards[user_columns[user_index].name].to_numpy()
        self.session_start_id_list = [0] + list(np.diff(user_ids).nonzero()[0] + 1)  # TODO: Use all interaction data of one user as a session (sequence)!
        self.all_users = np.unique(user_ids)

        user_data_list = [df_user.reset_index()[column.name].to_numpy().reshape(-1, 1) for column in user_columns]
        item_data_list = [df_item.reset_index()[column.name].to_numpy().reshape(-1, 1) for column in item_columns]

        self.user_numpy = np.concatenate(user_data_list, axis=-1).astype(np.int64)
        self.item_numpy = np.concatenate(item_data_list, axis=-1).astype(np.float32)

        self.cate_col = user_columns + item_columns
        self.num_col = reward_columns


    def compile(self, agent, reward_type="bce"):

        self.dones = np.diff(np.concatenate([self.session_numpy, self.session_numpy[-1:]+1]))
        # aa = np.arange(len(self.session_numpy))[self.dones.astype(bool)]

        for idx in tqdm(range(len(self.session_numpy)), total=len(self.session_numpy), desc="construct buffer data..."):
            state = np.concatenate([self.categorical_numpy[idx], self.numerical_numpy[idx]])
            # state_num = self.numerical_numpy[idx]
            action = np.random.rand(2)
            if self.dones[idx]:
                done = True
                nstate = state
                # nstate_num = state_num
            else:
                done = False
                nstate = np.concatenate([self.categorical_numpy[idx + 1], self.numerical_numpy[idx + 1]])
                # nstate_num = self.numerical_numpy[idx + 1]

            label = self.reward_numpy[idx]

            cvaction = np.array([action[0], action[1]])


            reward = -np.abs(cvaction - label)
            if reward_type == "bce":
                reward = label * np.log(np.clip(cvaction, 1e-4, 1)) + (1 - label) * np.log(np.clip(1 - cvaction, 1e-4, 1))

            transition = dict(
                state=state,
                # state_num=state_num,
                action=action,
                nstate=nstate,
                # nstate_num=nstate_num,
                reward=reward,
                done=done,
                label=label
            )
            agent.memory.store(**transition)
        print("buffer size:", len(agent.memory))

    def __getitem__(self, idx):
        cat_data = self.categorical_numpy[idx]
        num_data = self.numerical_numpy[idx]
        rwd_data = self.reward_numpy[idx]
        session_data = self.session_numpy[idx]
        return session_data, cat_data, num_data, rwd_data

    def __len__(self):
        return len(self.df_seq_rewards)


def get_datasets(args):
    DataClass = get_DataClass(args.env)
    dataset = DataClass()
    df_data, df_user, df_item, list_feat = dataset.get_data()

    user_features, item_features, reward_features = DataClass.get_features(df_user, df_item, args.use_userinfo)

    # NOTE: data augmentation
    if args.augment_type == 'mat':
        print("Augment Matrix Data")
        df_data_augmented = dataset.augment_matrix(df_data, df_item, time_field_name=dataset.time_field_name,
                                        augment_rate=args.augment_rate, strategies=args.augment_strategies)
    else:
        df_data_augmented = df_data

    user_columns, item_columns, reward_columns = get_xy_columns(
        df_user, df_item, user_features, item_features, reward_features, args.embed_dim, args.embed_dim)

    seq_columns = item_columns
    x_columns = user_columns

    df_seq_rewards, hist_seq_dict, to_go_seq_dict = dataset.get_and_save_seq_data(df_data_augmented, df_user, df_item,
        x_columns, reward_columns, seq_columns, args.max_item_list_len, args.len_reward_to_go, args.reload,
        dataset.time_field_name, args.augment_type, args.augment_rate, args.augment_strategies)


    item_features["dense"] = list(set(item_features["dense"]) - set(["rating"])) # Todo: for all multi-task SL methods
    item_columns = [col for col in item_columns if col.name != "rating"] # Todo: for all multi-task SL methods

    train_interaction_idx, test_interaction_idx = get_train_test_idx(df_seq_rewards, args.len_reward_to_go)
    df_test_rewards = df_seq_rewards.loc[test_interaction_idx]
    df_train_rewards = df_seq_rewards.loc[train_interaction_idx]

    train_dataset = RL_Dataset(df_train_rewards, df_user, df_item, user_columns, item_columns, reward_columns)
    test_dataset = RL_Dataset(df_test_rewards, df_user, df_item, user_columns, item_columns, reward_columns)

    # EnvClass = DataClass.get_env_class()
    # env = EnvClass(df_seq_rewards, target_features=reward_features, pareto_reload=args.reload, seq_columns=seq_columns)
    env = SLEnv(item_columns, reward_columns)
    env.compile_test(df_data, df_user, df_item)
    mat = dataset.get_completed_data(args.device)

    action_dim = len(reward_columns)
    field_dims = np.array([column.vocabulary_size for column in train_dataset.categorical_columns])
    return train_dataset, test_dataset, env, mat, action_dim, field_dims


def create_sub_agent(device, action_dim, cate_dim, num_numfeat, actor_name, agentcls, hyparams):
    hyparams.memory_size = 500000
    hyparams.init_episode = 200000

    # save_dir = os.path.join("saved_models", args.env, actor_name, "chkpt")
    # save_path = os.path.join(save_dir, "chkpt")
    hyparams.memory_path = "pretrain/memory.pkl"
    hyparams.pretrain_path = os.path.join("saved_models", args.env, actor_name, "chkpt")
    hyparams.init_training_step = 1000
    hyparams.actor_reg = 0
    hyparams.critic_lr = 1e-3
    hyparams.ips = False

    agent = agentcls(device, action_dim, cate_dim, num_numfeat, actor_name, hyparams)
    hyparams.set_curpath(str(agent) + "_" + actor_name)

    return agent


def main(task_num, expert_num, model_name, epoch, learning_rate, batch_size, embed_dim, weight_decay, device, args):
    device = torch.device(device)
    # 装载数据集

    train_dataset, test_dataset, env, mat, action_dim, field_dims = get_datasets(args)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    hyparams = Arguments()
    # 1. RL environment
    hyparams.train_rows = 500
    num_numfeat = len(test_dataset.numerical_columns) if len(test_dataset.numerical_columns) > 0 else 1
    agent = create_sub_agent(device, action_dim, field_dims, num_numfeat, args.actor_model_name, TD3_ESMMBCAgent, hyparams)  # TD3BC agent, 0 as ac loss by default
    polisher = RlLossPolisher(agent, device, action_dim, field_dims, num_numfeat, model_name, lambda_=1)

    train_dataset.compile(agent)

    collector = Collector_Baseline_SL(env, agent.actor, test_dataset, mat, args.len_reward_to_go, device=args.device)

    save_dir = os.path.join("saved_models", args.env, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)




    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=agent.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("memory size:", agent.memory.size)
    print("epoch | score | q_loss | ac_loss | a_loss | time")
    for epoch_i in range(0, epoch + 1):
        if epoch_i > 0:
            critic_loss, critic_loss2, actor_loss1, actor_loss2 = one_iter_offline(agent)
            sltrain(agent.actor, optimizer, train_data_loader, criterion, device, polisher)

        res = collector.collect()
        logzero.logger.info(f"Epoch: [{epoch_i}/{epoch}], Info: [{res}]")

    # save the pytorch model:
    save_dir = os.path.join("saved_models", args.env, f"{args.model_name}_{args.actor_model_name}", "chkpt")
    save_path = os.path.join(save_dir, f"RMTL.pt")
    torch.save(agent.state_dict(), save_path)




if __name__ == '__main__':
    args = get_SL_args()

    try:
        main(args.task_num, args.expert_num, args.model_name, args.epoch, args.learning_rate,
             args.batch_size, args.embed_dim, args.weight_decay, args.device, args)
    except Exception as e:
        var = traceback.format_exc()
        logzero.logger.error(var)

