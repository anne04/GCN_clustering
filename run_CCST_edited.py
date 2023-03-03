import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import pylab as pl
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import sparse
#from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import time

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument( '--data_type', default='nsc', help='"sc" or "nsc", \
        refers to single cell resolution datasets(e.g. MERFISH) and \
        non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help='input the preprocessed dataname that has to be fetched')
    parser.add_argument( '--lambda_I', type=float, default=0.8) #0.8 on MERFISH, 0.3 on ST
    parser.add_argument( '--meu', type=float, default=0.2)
    parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--model_path', type=str, default='model/')
    parser.add_argument( '--embedding_data_path', type=str, default='embedding_data/')
#    parser.add_argument( '--pca', type=int, default=0)
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')
    parser.add_argument( '--retrain', type=int, default=0 , help='Run which clustering at the end')
    parser.add_argument( '--model_load_path', type=str, default='model/')
    parser.add_argument( '--model_name', type=str, default='r1')
    parser.add_argument( '--training_data', type=str, default='provide please')
    parser.add_argument( '--GNN_type', type=str, default='GCNConv')


    args = parser.parse_args()

    args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/'
    args.model_path = args.model_path +'/'+ args.data_name +'/'
    #args.result_path = args.result_path +'/'+ args.data_name +'/'
    args.model_load_path = args.model_load_path +'/'+ args.data_name +'/'

    #print(args.model_name+', '+str(args.heads)+', '+args.training_data+', '+str(args.hidden) )

    start_time = time.time()
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    #args.result_path = args.result_path+'/'
    #if not os.path.exists(args.result_path):
    #    os.makedirs(args.result_path)
    print ('------------------------Model and Training Details--------------------------')
    print(args)
    from CCST_ST_utils_edited import CCST_on_ST
    CCST_on_ST(args)
    end_time = time.time() - start_time
    print('time elapsed %g min'%(end_time/60))


    '''
    if args.data_type == 'sc': # should input a single cell resolution dataset, e.g. MERFISH
        from CCST_merfish_utils import CCST_on_MERFISH
        CCST_on_MERFISH(args)
    elif args.data_type == 'nsc': # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
        from CCST_ST_utils_edited import CCC_on_ST
        CCST_on_ST(args)
        end_time = time.time() - start_time
        print('time elapsed %g min'%(end_time/60))
    else:
        print('Data type not specified')
     '''



