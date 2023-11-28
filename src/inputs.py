# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 9:58 下午
# @Author  : Chongming GAO
# @FileName: inputs.py

from collections import OrderedDict, defaultdict, namedtuple
from itertools import chain
import numpy as np
from torch import nn
import torch

# from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, varlen_embedding_lookup, \
#     get_varlen_pooling_list

DEFAULT_GROUP_NAME = "default_group"




class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'embedding_dim', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, embedding_dim=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, embedding_dim, dtype)

    def __hash__(self):
        return self.name.__hash__()

class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()
    
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

class SparseFeatP(SparseFeat):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        return super(SparseFeatP, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                               embedding_name, group_name)

    def __init__(self, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        self.padding_idx = padding_idx



def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            # TODO: add hash function
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0]:lookup_idx[1]].long())  # (lookup_idx)

    return varlen_embedding_vec_dict


def get_dataset_columns(dim_user, dim_action, num_user, num_action, envname="VirtualTB-v0"):
    user_columns, action_columns, feedback_columns = [], [], []
    has_user_embedding, has_action_embedding, has_feedback_embedding = None, None, None
    if envname == "VirtualTB-v0":
        user_columns = [DenseFeat("feat_user", 88)]
        action_columns = [DenseFeat("feat_item", 27)]
        # feedback_columns = [SparseFeat("feat_feedback", 11, embedding_dim=27)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = True
        has_action_embedding = True
        has_feedback_embedding = True
    else: # for kuairecenv, coat
        user_columns = [SparseFeatP("feat_user", num_user, embedding_dim=dim_user)]
        action_columns = [SparseFeatP("feat_item", num_action, embedding_dim=dim_action)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = False
        has_action_embedding = False
        has_feedback_embedding = True

    return user_columns, action_columns, feedback_columns, \
           has_user_embedding, has_action_embedding, has_feedback_embedding


def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    """
        Args:
            X: input Tensor [batch_size x hidden_dim]
            sparse_embedding_dict: nn.ModuleDict, {embedding_name: nn.Embedding}
            sparse_input_dict: OrderedDict, {feature_name:(start, start+dimension)}
            sparse_feature_columns: list, sparse features
            return_feat_list: list, names of feature to be returned, defualt () -> return all features
            mask_feat_list, list, names of feature to be masked in hash transform
        Return:
            group_embedding_dict: defaultdict(list)
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            # TODO: add hash function
            # if fc.use_hash:
            #     raise NotImplementedError("hash function is not implemented in this version!")
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    sparse_dict = {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse, padding_idx=feat.padding_idx)
                   for feat in sparse_feature_columns + varlen_sparse_feature_columns}
    dense_dict = {feat.name: nn.Sequential(nn.Linear(feat.dimension, feat.embedding_dim), nn.Tanh()) for feat in dense_feature_columns}
    all_dict = {**sparse_dict, **dense_dict}

    embedding_dict = nn.ModuleDict(all_dict)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.padding_idx is None:
                nn.init.normal_(module.weight, mean=0, std=init_std)
            else:
                nn.init.normal_(module.weight[:module.padding_idx], mean=0, std=init_std)
                nn.init.normal_(module.weight[module.padding_idx+1:], mean=0, std=init_std)
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)

    embedding_dict.apply(lambda x: _init_weights)
            
    return embedding_dict.to(device)


def compute_input_dim(feature_columns, include_sparse=True, include_dense=True, feature_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, (SparseFeatP, VarLenSparseFeat)), feature_columns)) if len(
        feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    dense_input_dim = sum(
        map(lambda x: x.embedding_dim, dense_feature_columns))
    if feature_group:
        sparse_input_dim = len(sparse_feature_columns)
    else:
        sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
    input_dim = 0
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim

def input_from_feature_columns(X, feature_columns, embedding_dict, feature_index, support_dense: bool, device):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]
    
    dense_list = [embedding_dict[feat.name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].unsqueeze(-1)) for feat in dense_feature_columns]

    # sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
    #                                               varlen_sparse_feature_columns)
    # varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
    #                                                        varlen_sparse_feature_columns, device)

    # dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
    #                     dense_feature_columns]
    

    return sparse_embedding_list + dense_list

def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
    
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError