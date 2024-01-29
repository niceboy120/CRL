import argparse
import os
import traceback

import logzero
import torch
import sys


sys.path.extend(["./src"])
from collector import Collector

sys.path.extend(["baselines/CDT4Rec"])
from baselines.CDT4Rec.cdt4rec import CDT4Rec, CDT4RecConfig


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

    parser.add_argument("--model_name", type=str, default="CDT4Rec")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--local_D", type=int, default=32)
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

    parser.add_argument("--cuda", type=int, default=2)
    parser.add_argument("--message", type=str, default="cdt4rec")

    # parser.add_argument("--env", type=str, default="KuaiRand-1K")
    args = parser.parse_known_args()[0]
    # args = parser.parse_args()

    if args.env not in ["ml-1m", "KuaiRand-Pure", "Zhihu-1M"]:
        parser.print_help()
        sys.exit(1)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{args.cuda}'
        # model = torch.nn.DataParallel(model).to(device)
    args.device = device

    return args



def main(args):
    args = get_common_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)
    # NOTE: set seed
    set_seed(args.seed)

    train_dataset, test_dataset, env, mat = prepare_dataset(args)
    # get model
    mconf = CDT4RecConfig(
        train_dataset.num_items, pos_drop=0.1, resid_drop=0.1, 
        local_num_feat=len(train_dataset.reward_columns) + 2,  # 1 dim for the sequence embedding and 1 dim for the user embedding
        x_columns=train_dataset.x_columns,  reward_columns=train_dataset.reward_columns, seq_columns=train_dataset.seq_columns, y_column=train_dataset.y_column,
        model_name=args.model_name, N_head=8, global_D=args.global_D, local_N_head=4, local_D=args.local_D,
        n_layer=args.n_layer, max_seqlens=args.max_item_list_len - args.len_reward_to_go, gru_layers=1, hidden_size=args.local_D, device=args.device,
        col_item=[col.name for col in train_dataset.seq_columns].index("item_id"),
    )
    model = CDT4Rec(mconf).to(args.device)

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

    # save the pytorch model:
    save_path = os.path.join(MODEL_SAVE_PATH, "chkpt", f"[{args.message}]_epoch[{args.epochs}].pt")
    torch.save(model.cpu().state_dict(), save_path)

    # When next time you use the saved model, just do:
    # model = CDT4Rec(mconf).to(args.device)
    # model.load_state_dict(torch.load(save_path))


if __name__ == "__main__":

    args = get_args()
    try:
        main(args)
    except Exception as e:
        var = traceback.format_exc()
        logzero.logger.error(var)