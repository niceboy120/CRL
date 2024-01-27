import argparse
import traceback

import logzero
import torch

import sys
sys.path.extend(["./src"])

from baselines.DT4Rec.dt4rec import DT4Rec
from collector import Collector
from ctrl import CTRL, CTRLConfig


from utils import prepare_dir_log, set_seed
# from inputs import SparseFeatP
from data import get_DataClass, get_common_args, get_datapath, prepare_dataset

from trainer import Trainer, TrainerConfig



# from starformer import Starformer, StarformerConfig


# For vscode debug!!
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: "<Tensor shape={}, device={}, dtype={}, value=\n{}>".format(tuple(self.shape), self.device, self.dtype, original_repr(self))
# torch.Tensor.__repr__ = lambda self: "<Tensor shape={}, device={}, dtype={}>".format(
#     tuple(self.shape), self.device, self.dtype
# )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ml-1m")
    # parser.add_argument("--env", type=str, default="zhihu-1m")
    parser.add_argument("--max_item_list_len", type=int, default=30)
    parser.add_argument("--len_reward_to_go", type=int, default=10)

    parser.add_argument("--model_name", type=str, default="ctrl")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--global_D", type=int, default=40)
    parser.add_argument("--n_layer", type=int, default=2)

    parser.add_argument("--is_reload", dest="reload", action="store_true")
    parser.add_argument("--no_reload", dest="reload", action="store_false")
    parser.set_defaults(reload=False)

    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--entity_dim", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--cuda_id", type=int, default=2)
    parser.add_argument("--message", type=str, default="test")

    # parser.add_argument("--env", type=str, default="KuaiRand-1K")
    args = parser.parse_known_args()[0]
    # args = parser.parse_args()

    if args.env not in ["ml-1m", "KuaiRand-Pure", "Zhihu-1M"]:
        parser.print_help()
        sys.exit(1)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{args.cuda_id}'
        # model = torch.nn.DataParallel(model).to(device)
    args.device = device

    return args



def main(args):
    args = get_common_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)
    # NOTE: set seed
    set_seed(args.seed)
    args.local_D = args.n_embd
    train_dataset, test_dataset, env, mat = prepare_dataset(args)
    # get model
    mconf = CTRLConfig(
        train_dataset.num_items,
        n_embd=args.n_embd,
        item_col_id=[column.name for column in train_dataset.seq_columns].index("item_id"),
        pos_drop=0.1, resid_drop=0.1,
        local_num_feat=len(train_dataset.reward_columns) + 2,  # 1 dim for the sequence embedding and 1 dim for the user embedding
        x_columns=train_dataset.x_columns,  reward_columns=train_dataset.reward_columns, seq_columns=train_dataset.seq_columns, y_column=train_dataset.y_column,
        model_name=args.model_name, N_head=8,  global_D=args.global_D, local_N_head=4, local_D=args.local_D,
        n_layer=args.n_layer, max_seqlens=args.max_item_list_len - args.len_reward_to_go, gru_layers=1, device=args.device,
    )
    model = DT4Rec(mconf).to(args.device)

    tconf = TrainerConfig(
        max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, num_items=train_dataset.num_items,
        lr_decay=True, warmup_tokens=512 * 20,
        final_tokens=10 * len(train_dataset) * args.max_item_list_len - args.len_reward_to_go,
        num_workers=8, seed=args.seed, model_name=args.model_name, game=args.env,
        max_seqlens=args.max_item_list_len - args.len_reward_to_go, 
        args=args,
    )

    collector = Collector(env, model, test_dataset, mat, args.len_reward_to_go)
    
    trainer = Trainer(model, collector, train_dataset, test_dataset, tconf)
    trainer.train()
    


if __name__ == "__main__":

    args = get_args()
    try:
        main(args)
    except Exception as e:
        var = traceback.format_exc()
        logzero.logger.error(var)