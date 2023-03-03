import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
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


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def PCA_process(X, nps):
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC

def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    #if merge:
    #    cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')

    return cluster_labels, score



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, help='data name?')
    parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--embedding_data_path', type=str, default='embedding_data/')
    parser.add_argument( '--result_path', type=str, default='result/')
    parser.add_argument( '--input_type', type=str, default='gene_exp') 
    parser.add_argument( '--PCA', type=int, default=0, help='run PCA or not')
    parser.add_argument( '--pca_count', type=int, default=200, help='number of pca')
    parser.add_argument( '--model_name', type=str, help='model name')
    parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MER$
    parser.add_argument( '--cluster_alg', type=str, default='kmeans' , help='Run which clustering at the end')
    

    args = parser.parse_args()

    barcode_file = args.data_path +'/'+ args.data_name +'/' + 'barcodes.npy'
    result_path = args.result_path +'/'+ args.data_name +'/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

#    print(args)

    cluster_type = args.cluster_alg # 'leiden' #'kmeans' # 'louvain' # 'leiden'
    n_clusters = args.n_clusters

    if args.input_type == 'gene_exp':
        print('Using only gene expression for clustering')
        data_file = args.data_path + args.data_name +'/'
        X_data = np.load(data_file + 'features.npy')
        num_cells = X_data.shape[0]
        print('number of cells = %d'%num_cells)
        if args.PCA == 1:
            X_input = PCA_process(X_data, nps=args.pca_count)
        else:
            X_input = X_data    


    elif args.input_type == 'embedding':
        args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/' + args.model_name + '_'
        X_embedding_filename =  args.embedding_data_path+'Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)

        num_cells = X_embedding.shape[0]
        print('number of cells = %d'%num_cells)

        if args.PCA == 1:  
            X_input = PCA_process(X_embedding, nps=args.pca_count)
        else:
            X_input =  X_embedding



    print("-----------Clustering-------------")
    if cluster_type == 'kmeans':
        print('Shape of data to cluster:', X_input.shape)
        cluster_labels, score = Kmeans_cluster(X_input, n_clusters)
    else:
	adata = ad.AnnData(X_input)
        sc.pp.neighbors(adata, knn=False, method='gauss', use_rep='X', n_neighbors=15) 
        #sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        #sc.pp.neighbors(adata, knn=False, method='gauss', n_neighbors=15, n_pcs=50) 
        #sc.pp.neighbors(adata, knn=False, method='gauss', use_rep='X', n_neighbors=15)

        if cluster_type == 'leiden':
            print('leiden start')
            sc.tl.leiden(adata, directed=False, key_added="leiden", resolution=3.0)
            cluster_labels = np.array(adata.obs['leiden'])
            print ('leiden done')
        if cluster_type == 'louvain':
            print('louvain start')
            sc.tl.louvain(adata, key_added="louvain", resolution=1)
            cluster_labels = np.array(adata.obs['louvain'])
            print('louvain done')

            #sc.tl.umap(adata)
            #sc.pl.umap(adata, color=['leiden'], save='_lambdaI_' + str(lambda_I) + '.png')

    
    barcode_info = np.load(barcode_file, allow_pickle=True)
    all_data = []
    for index in range (0, num_cells):
        all_data.append([barcode_info[index], cluster_labels[index]])

#filename = args.result_path+'/'+ args.cluster_alg  +'_barcode_label_node_embedding.csv'
    filename = result_path+'/'+args.data_name +'_'+ args.cluster_alg +'_' + args.input_type +'_barcode_label.csv' #_node_embedding.csv'
    print('saving the clustering result here: ' + filename)
    f = open(filename, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    # write the header
    writer.writerow(['label','item'])
    writer.writerows(all_data)
    f.close()



