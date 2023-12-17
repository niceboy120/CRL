import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

from environments.ML_1M.ml1m_env import ML1MEnv
from environments.base_data import BaseData
from inputs import SparseFeatP, DenseFeat
# from ..utils import get_novelty, get_serendipity, get_diversiy


ENVPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ENVPATH, "data")
FIGPATH = os.path.join(ENVPATH, "figs")
RESULTPATH = os.path.join(ENVPATH, "data_processed")


def load_data():
    # load dataset from files:
    # 读取 inter_impression.csv
    df_inter_impression = pd.read_csv(os.path.join(DATAPATH, "inter_impression.csv"), header=None,
                                      names=["user_id", "answer_id", "impression timestamp", "click timestamp"])

    # 读取 info_user.csv
    df_info_user = pd.read_csv(os.path.join(DATAPATH, "info_user.csv"), header=None, sep=",",
                               names=["user_id", "register_timestamp", "gender", "login_frequency",
                                      "num_followers", "num_topics_followed", "num_questions_followed",
                                      "num_answers", "num_questions", "num_comments", "num_thanks_received",
                                      "num_comments_received", "num_likes_received", "num_dislikes_received",
                                      "register_type", "register_platform", "from_android", "from_iphone",
                                      "from_ipad", "from_pc", "from_mobile_web", "device_model", "device_brand",
                                      "platform", "province", "city", "topic_IDs_followed"])

    # 读取 inter_query.csv
    df_inter_query = pd.read_csv(os.path.join(DATAPATH, "inter_query.csv"), header=None,
                                 names=["user_id", "token_id", "query_timestamp"])

    # 读取 info_answer.csv
    df_info_answer = pd.read_csv(os.path.join(DATAPATH, "info_answer.csv"), header=None, sep=",",
                                 names=["answer_id", "question_id", "is_anonymous", "author_id", "is_high_value",
                                        "recommended_by_the_editor_or_not", "creation_timestamp", "contain_pictures",
                                        "contain_videos", "num_thanks", "num_likes", "num_comments", "num_collections",
                                        "num_dislikes", "num_reported", "num_helpless", "token_IDs", "topic_IDs"])

    # 读取 info_question.csv
    df_info_question = pd.read_csv(os.path.join(DATAPATH, "info_question.csv"), header=None, sep=",",
                                   names=["question_id", "timestamp", "num_answers", "num_followers", "num_invitations",
                                          "num_comments", "token_IDs", "topic_IDs"])

    # 读取 info_author.csv
    df_info_author = pd.read_csv(os.path.join(DATAPATH, "info_author.csv"), header=None,
                                 names=["author_id", "is_excellent_author", "num_followers", "is_excellent_answerer"])

    # 读取 info_topic.csv
    df_info_topic = pd.read_csv(os.path.join(DATAPATH, "info_topic.csv"), header=None, names=["topic_id"])

    return df_inter_impression, df_info_user, df_inter_query, df_info_answer, df_info_question, df_info_author, df_info_topic

def main():
    df_inter_impression, df_info_user, df_inter_query, df_info_answer, df_info_question, df_info_author, df_info_topic = load_data()
    df_inter_impression["answer_id"].nunique()
    df_info_answer["answer_id"].nunique()
    df_info_answer["answer_id"].nunique()


if __name__ == "__main__":
    main()


