import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import numpy as np
from GraphST import GraphST

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


file_fold='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/' #please replace 'file_fold' with the datapath
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()
adata

# load the GraphST model and train it on our data
model = GraphST(adata, device=device)
adata = model.train()

# 'emb' is the learned representation or embedding that we need
X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/data/GraphST_embedding_saved/V1_Human_Lymph_Node_spatial/V1_Human_Lymph_Node_spatial_GraphST_embedding.npy'
X_embedding = adata.obsm['emb'] 
np.save(X_embedding_filename, X_embedding)

file_list = ['/cluster/projects/schwartzgroup/deisha/data/spaceranger/10x_dlpfc/outs',
             
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp2/exp2_A1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp2/exp2_B1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp2/exp2_C1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp2/exp2_D1/outs',
             
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp3/exp3_A1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp3/exp3_B1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp3/exp3_C1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp3/exp3_D1/outs',
             
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp4/exp4_A1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp4/exp4_B1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp4/exp4_C1/outs',
             '/cluster/projects/schwartzgroup/deisha/data/spaceranger/exp4/exp4_D1/outs',
            
             '/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_A1_PDA_128033_Pa_P_Spatial10x/outs',
             '/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_B1_PDA_134275_Pa_R_Spatial10x/outs',
             '/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_C1_PDA_140694_Pa_P_Spatial10x/outs',
             '/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_D1_PDA_130355_Pa_R_Spatial10x/outs'
            ]

X_embedding_filename = ['10x_dlpfc', 
                        'exp2_A1', 'exp2_B1', 'exp2_C1', 'exp2_D1', 
                        'exp3_A1', 'exp3_B1', 'exp3_C1', 'exp3_D1', 
                       'exp4_A1', 'exp4_B1', 'exp4_C1', 'exp4_D1',
                        'exp1_A1', 'exp1_B1', 'exp1_C1', 'exp1_D1',                       
                       ]

for index in range (5, 13): #len(file_list)):
    sample_name = file_list[index]
    print(sample_name)
    adata = sc.read_visium(sample_name, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata

    # load the GraphST model and train it on our data
    model = GraphST(adata, device=device)
    adata = model.train()

    # 'emb' is the learned representation or embedding that we need
    X_embedding_save = '/cluster/projects/schwartzgroup/fatema/data/GraphST_embedding_saved/'+X_embedding_filename[index]+'_X_embedding.npy'
    X_embedding = adata.obsm['emb'] 
    np.save(X_embedding_save, X_embedding)

  
