import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group
import qnorm
#from sklearn.preprocessing import quantile_transform
import pickle
from scipy import sparse
import pickle
import scipy.linalg
from sklearn.metrics.pairwise import euclidean_distances
####################  get the whole training dataset


#rootPath = os.path.dirname(sys.path[0])
#os.chdir(rootPath+'/CCST')

print("libraries imported")

def main(args):
    print("reading raw data from "+args.data_path)

    data_fold = args.data_path #+args.data_name+'/'
    print(data_fold)

    generated_data_fold = args.generated_data_path + args.data_name+'/'
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)

    adata_h5 = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)

    gene_ids = adata_h5.var['gene_ids']
    coordinates = adata_h5.obsm['spatial']
    cell_barcode = np.array(adata_h5.obs.index)


    print('===== Preprocessing Data ')
    sc.pp.filter_genes(adata_h5, min_cells=args.min_cells)
#    temp = qnorm.quantile_normalize(np.transpose(scipy.sparse.csr_matrix.toarray(adata_h5.X)))  #quantile_transform(scipy.sparse.csr_matrix.toarray(adata_h5.X), copy=True)
#    adata_X = np.transpose(temp)

#    adata_X = scipy.sparse.csr_matrix(adata_X)
#    adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, inplace=False)['X']
    adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)

    if args.pca>0:
        print('doing PCA')
        adata_X = sc.pp.pca(adata_X, n_comps=args.pca)
    print('PCA done')

    features = adata_X

    print('dumping cell_VS_gene_exp here: '+generated_data_fold + 'features.npy')
    # pickle format 
    with open(generated_data_fold + 'features', 'wb') as fp:
        pickle.dump(features, fp)
    # .npy format
    np.save(generated_data_fold + 'features.npy', features)


    coordinates = np.array(coordinates)
    print('dumping cell coordinates here: '+generated_data_fold + 'coordinates.npy')
    np.save(generated_data_fold + 'coordinates.npy', coordinates)

    print('dumping cell barcodes here: '+generated_data_fold + 'barcodes.npy')
    cell_barcode = np.array(cell_barcode)
    np.save(generated_data_fold + 'barcodes.npy', cell_barcode)
    
    # 
    #coordinates = np.load(generated_data_fold + 'coordinates.npy')
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(coordinates, coordinates)

    #from sklearn.metrics.pairwise import manhattan_distances
    #distance_matrix = manhattan_distances(coordinates, coordinates)

    '''for threshold in [300]:#range (210,211):#(100,400,40):
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(cell_num*2))) #300 22064 2.9046866771985256
        #threshold=2000
        #np.where(distance_matrix<threshold)[0].shape[0] # these are the number of the edges in the adj matrix
        #416332
     '''


    

    threshold=300
    distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
    #distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
    for i in range(distance_matrix_threshold_I.shape[0]):
        for j in range(distance_matrix_threshold_I.shape[1]):
            if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                distance_matrix_threshold_I[i,j] = 1
                #distance_matrix_threshold_W[i,j] = distance_matrix[i,j]

    
    distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) ## do not normalize adjcent matrix
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
    print('dumping cell_VS_cells adjacency matrix here: '+generated_data_fold + 'Adjacent')
    with open(generated_data_fold + 'Adjacent', 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_N_crs, fp)

    '''
        threshold=2000
        for i in range(distance_matrix.shape[0]):
            max_value=np.max(distance_matrix[i,:])
            for j in range(distance_matrix.shape[1]):
                if distance_matrix[i,j] > threshold: # and distance_matrix[i,j] >= 0:
                    distance_matrix[i,j] = max_value

            min_value=np.min(distance_matrix[i,:])
            print('min_value: ',min_value)
            for j in range(distance_matrix.shape[1]):
                distance_matrix[i,j]=1-(distance_matrix[i,j]-min_value)/(max_value-min_value)

        ############### get normalized sparse adjacent matrix
        distance_matrix = np.float32(distance_matrix) ## do not normalize adjcent matrix
        distance_matrix_crs = sparse.csr_matrix(distance_matrix)
        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_crs, fp)
    '''
    '''
     elif args.all_distance == 1:

        for i in range (0,distance_matrix.shape[0]):
            distance_matrix_min=np.min(distance_matrix[i,:])
            distance_matrix_max=np.max(distance_matrix[i,:])
            distance_matrix[i]=1-(distance_matrix[i,:]-distance_matrix_min)/(distance_matrix_max-distance_matrix_min)


        distance_matrix_crs = sparse.csr_matrix(distance_matrix)
        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_crs, fp)
    '''
    print("data preprocess done")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--min_cells', type=float, default=1, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    parser.add_argument( '--data_path', type=str, help='The path to dataset')
    parser.add_argument( '--data_name', type=str, help='The name of dataset')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
    parser.add_argument( '--pca', type=int, default=0, help='PCA count')

    #parser.add_argument( '--all_distance', type=int, default=0, help='The folder to store the generated data')
    args = parser.parse_args()

    main(args)
