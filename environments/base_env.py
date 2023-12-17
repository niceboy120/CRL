import os
import pickle

import logzero
import numpy as np

import pandas as pd
from tqdm import tqdm
from numba import njit


from .get_pareto_front import get_kde_density, get_pareto_front


class BaseEnv:
    def __init__(self, df_seq_rewards, target_features, bin=100, percentile=90, pareto_reload=False):

        self.target_features = target_features
        pareto_front_path = f"pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        raw_pareto_front_path = f"raw_pareto_front_bin{bin}_per{percentile}_({','.join(target_features)}).pkl"
        pareto_front_filepath = os.path.join(self.RESULTPATH, pareto_front_path)
        raw_pareto_front_filepath = os.path.join(self.RESULTPATH, raw_pareto_front_path)

        # if not reload and os.path.exists(pareto_front_filepath):
        #     print("pareto front has already been computed! Loading...")
        #     with open(pareto_front_filepath, "rb") as f:
        #         self.pareto_front = pickle.load(f)
        #     print("pareto front loaded!")
        #     return self.pareto_front
        if not pareto_reload and os.path.exists(raw_pareto_front_filepath): # use raw pareto front
            print("raw pareto front has already been computed! Loading...")
            with open(raw_pareto_front_filepath, "rb") as f:
                self.pareto_front = pickle.load(f)
            print("raw pareto front loaded!")
            # return self.raw_pareto_front
        else:
            density_flat, grid_flat = get_kde_density(df_seq_rewards, target_features, bin)
            self.pareto_front = get_pareto_front(density_flat, grid_flat, percentile)
            print(f"pareto front for bin:{bin} and percentile:{percentile} is ", self.pareto_front)
            # pickle dump pareto_front
            with open(pareto_front_filepath, "wb") as f:
                pickle.dump(self.pareto_front, f)
            
            # get raw pareto front
            key_statistics = df_seq_rewards.describe()[target_features]
            self.pareto_front = (key_statistics.loc["max"] - key_statistics.loc["min"]).to_numpy() * self.pareto_front + key_statistics.loc["min"].to_numpy()
            print(f"Raw pareto front for bin:{bin} and percentile:{percentile} is ", self.pareto_front)

            
            # pickle dump pareto_front
            with open(raw_pareto_front_filepath, "wb") as f:
                pickle.dump(self.pareto_front, f)

        logzero.logger.info(f"All Pareto Fronts for Env are: {self.pareto_front}")

    def reset(self, ):
        pass

    def step(self, buffer):
        pass
        
