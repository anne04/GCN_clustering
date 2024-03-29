import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import os

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, help='The name of dataset')
    #parser.add_argument( '--cell_label_file_name', type=str, default='kmeans_gene_exp_barcode_label.csv', help='data name?')
    parser.add_argument( '--embedding_data_path', type=str, default='embedding_data/', help='data path')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--model_name', type=str, help='input the name of the model that is used for node embedding generation')
    parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()
    
    #args.data_name = 'exp2_V10M25_61_D1_64630_Spatial10X' 
    #args.model_name = 'exp2_V10M25_61_D1_64630_Spatial10X_test1' 
    cell_barcode = np.load(args.generated_data_path + args.data_name+'/'+'barcodes.npy', allow_pickle=True)
    barcode_info=[]
    barcode_info.append('')
    for i in range (0, cell_barcode.shape[0]):
        barcode_info.append(cell_barcode[i])

##############################################

    X_embedding_filename = '/home/fatema/scratch/best_gmi_X_embedding_exp1_C1.npy' #args.embedding_data_path + args.data_name + '/' + args.model_name + '_Embed_X.npy'
    X_embedding = np.load(X_embedding_filename, allow_pickle=True)
    num_feature = X_embedding.shape[1] # row = spots, collumns = features

#####################
    feature_id = np.zeros((1, num_feature))
    for i in range (0, num_feature):
        feature_id[0,i] = i+1

    X_embedding = np.concatenate((feature_id, X_embedding)) # first column should have some sort of ID. I am using just unique integer IDs. 
    X_embedding_T = np.transpose(X_embedding) # To match with the too-many-cells input format
    
    toomanycells_input_filename = args.result_path +'/'+ args.data_name +'/' + args.model_name  + '_node_embedding_GMI.csv'
    #toomanycells_input_filename = args.result_path +'/'+ args.data_name +'/' + args.model_name  + '_node_embedding.csv'
    if not os.path.exists(args.result_path +'/'+ args.data_name):
        os.makedirs(args.result_path +'/'+ args.data_name)

    f=open(toomanycells_input_filename, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    # write the header
    writer.writerow(barcode_info)
    writer.writerows(X_embedding_T)
    f.close()
    print('preprocess done')
####################################################


