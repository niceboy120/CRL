import logzero
from logzero import logger
import os
import time
import datetime
import json
import socket


def set_seed(seed=42):
    import numpy as np
    import torch
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    
def log_config(args):
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)
    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)

def create_dir(create_dirs):
    """
    create necessary dirs.
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))

def prepare_dir_log(args):
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.model_name)
    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    hostname = socket.gethostname()
    args.hostname = hostname
    logger.info(json.dumps(vars(args), indent=2))

    return MODEL_SAVE_PATH, logger_path
