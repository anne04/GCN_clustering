import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
import gzip
import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax
from torch_geometric.data import Data, DataLoader

from CCST_edited import get_graph, train_DGI


def get_data(args):

    data_file = args.data_path + args.data_name +'/'

    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)

    X_data = np.load(data_file + 'features.npy')


    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)


    adj = adj_0-adj_I # diagonal becomes zero
    print('spatial_impact:', args.meu)
    adj = adj*args.meu + adj_I*args.lambda_I # adj*0.05 + adj_I # 2k

    #cell_type_indeces = np.load(data_file + 'cell_types.npy')
    
    return adj_0, adj, X_data, 5 #cell_type_indeces



def CCST_on_ST(args):

    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    adj_0, adj, X_data, cell_type_indeces = get_data(args)

    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)

    if args.DGI and (lambda_I>=0):
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list, batch_size=batch_size)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)

        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename =  args.embedding_data_path + args.model_name + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)
    print("DGI is finished")

