import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import pickle
import os
import sys
import time

# def fit_kde(x, bandwidth=0.2, **kwargs):


def is_pareto_efficient(costs, return_mask=True):
    """
    Source code from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_kde_density(df_seq_rewards, target_columns, bin=100):
    
    print(df_seq_rewards.head())

    df = df_seq_rewards[target_columns]
    normed_df = (df - df.min()) / (df.max() - df.min())

    points = normed_df.to_numpy().T
    kernel = stats.gaussian_kde(points)

    grid = eval("np.mgrid[" + "0:1:{}j, ".format(bin) * len(target_columns) + "]")

    grid_flat = np.vstack(list(map(np.ravel, grid)))

    print("computing kernel density...")
    t = time.time()
    density_flat = kernel(grid_flat)
    t = time.time() - t
    print("kernel density computed in {t} seconds".format(t=t))

    return density_flat, grid_flat


def get_pareto_front(density_flat, grid_flat, percentile=90):
    threshold = np.percentile(density_flat, percentile)
    print(threshold)

    all_indices = grid_flat[:, density_flat >= threshold]

    pareto_index = is_pareto_efficient(-all_indices.T)

    pareto_front = all_indices.T[pareto_index]

    return pareto_front



def main(dataset, bin, percentile):
    pareto_front_path = f"pareto_front_bin{bin}_per{percentile}.pkl"

    if dataset == "ml-1m":
        df_seq_filepath = "./ML_1M/df_seq_rewards.csv"
        pareto_front_filepath = os.path.join("./ML_1M", pareto_front_path)

        target_columns = ["novelty", "diversity", "serendipity_4", "sum_rating"]
        target_columns = ["novelty", "diversity", "sum_rating"]

    elif dataset == "KuaiRand-1K":
        df_seq_filepath = "./KuaiRand-1K/data/df_seq_rewards.csv"
        pareto_front_filepath = os.path.join("./KuaiRand-1K/data/", pareto_front_path)

        target_columns = ["novelty", "diversity", "sum_is_click"]
        target_columns = ["novelty", "diversity", "serendipity_click", "sum_is_click"]

    df_seq_rewards = pd.read_csv(df_seq_filepath)
    density_flat, grid_flat = get_kde_density(df_seq_rewards, target_columns, bin)
    pareto_front = get_pareto_front(density_flat, grid_flat, percentile)
    
    print(f"pareto front for bin:{bin} and percentile:{percentile} is ", pareto_front)
    
    # pickle dump pareto_front
    with open(pareto_front_filepath, "wb") as f:
        pickle.dump(pareto_front, f)

    return pareto_front

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--bin", type=int, default=100)
    parser.add_argument("--percentile", type=int, default=90)
    # args = parser.parse_known_args()[0]
    args = parser.parse_args()


    if args.dataset not in ["ml-1m", "KuaiRand-1K"]:
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == "__main__":

    args = get_args()

    main(dataset = args.dataset, bin=args.bin, percentile=args.percentile)
