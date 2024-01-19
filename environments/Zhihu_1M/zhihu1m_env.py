import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

from inputs import SparseFeatP
from ..utils import get_novelty, get_serendipity, get_diversiy
from ..base_env import BaseEnv


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")

class Zhihu1MEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self.RESULTPATH = RESULTPATH
        self.FIGPATH = FIGPATH
        self.DATAPATH = DATAPATH

        super().__init__(*args, **kwargs)



