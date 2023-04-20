
# edited from: https://github.com/zpeng27/GMI/blob/master/execute.py
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy.sparse as sp
from models import GMI, LogReg
from utils import process
import pickle
import gzip
"""command-line interface"""
parser = argparse.ArgumentParser(description="PyTorch Implementation of GMI")
parser.add_argument('--dataset', default='cora',
                    help='name of dataset. if on citeseer and pubmed, the encoder is 1-layer GCN. you need to modify gmi.py')
parser.add_argument('--gpu', type=int, default=0,
                    help='set GPU')
"""training params"""
parser.add_argument('--hid_units', type=int, default=512,
                    help='dim of node embedding (default: 512)')
parser.add_argument('--nb_epochs', type=int, default=550,
                    help='number of epochs to train (default: 550)')
parser.add_argument('--epoch_flag', type=int, default=20,
                    help=' early stopping (default: 20)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--l2_coef', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--negative_num', type=int, default=5,
                    help='number of negative examples used in the discriminator (default: 5)')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='parameter for I(w_ij; a_ij) (default: 1.0)')
parser.add_argument('--activation', default='prelu',
                    help='activation function')
parser.add_argument('--meu', default=0.2,
                    help='spatial impact')
parser.add_argument('--lambda_I', default=0.8,
                    help='own gene expression impact')

###############################################
# This section of code adapted from Petar Veličković/DGI #
###############################################

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
#data_file = 'exp1_V10M25_60_C1_140694_Spatial10X/' #'/project/def-gregorys/fatema/GCN_clustering/generated_data_pca/exp1_V10M25_60_C1_140694_Spatial10X/'
data_file = '/project/def-gregorys/fatema/GCN_clustering/generated_data_pca/exp1_V10M25_60_C1_140694_Spatial10X/'
print('Loading ', args.dataset)
adj_ori, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
features, _ = process.preprocess_features(features)
# adj_ori <2708x2708 sparse array of type '<class 'numpy.int64'>' with 10556 stored elements in Compressed Sparse Row format>
# feature.shape (2708, 1433)


with open(data_file + 'Adjacent', 'rb') as fp:
    adj_ori = pickle.load(fp) 

#<2298x2298 sparse matrix of type '<class 'numpy.float32'>' with 13318 stored elements in Compressed Sparse Row format>
# will change this adj_ori matrix so that weight is reflected

features = np.load(data_file + 'features.npy')

#########
num_points = features.shape[0]
adj_I = np.eye(num_points)
adj_I = sp.csr_matrix(adj_I)

adj_ori = adj_ori-adj_I # diagonal becomes zero
print('spatial_impact:', args.meu)
adj_ori = adj_ori*args.meu + adj_I*args.lambda_I #
#########

adj = process.normalize_adj(adj_ori) 
sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)

adj_dense = adj_ori.toarray()
adj_target = adj_dense
adj_row_avg = 1.0/np.sum(adj_dense, axis=1) # columns are merged
adj_row_avg[np.isnan(adj_row_avg)] = 0.0
adj_row_avg[np.isinf(adj_row_avg)] = 0.0
adj_dense = adj_dense*1.0 
for i in range(adj_ori.shape[0]):
    adj_dense[i] = adj_dense[i]*adj_row_avg[i]
adj_ori = sp.csr_matrix(adj_dense, dtype=np.float32)



nb_nodes = features.shape[0]
ft_size = features.shape[1]
#nb_classes = labels.shape[1]

features = torch.FloatTensor(features[np.newaxis])

#labels = torch.FloatTensor(labels[np.newaxis])
#idx_train = torch.LongTensor(idx_train)
#idx_val = torch.LongTensor(idx_val)
#idx_test = torch.LongTensor(idx_test)
if torch.cuda.is_available():
    print('GPU available: Using CUDA')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model = GMI(ft_size, args.hid_units, args.activation).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
'''
    model.cuda()
    features = features.cuda()
    sp_adj = sp_adj.cuda()
'''
features = features.to(device)
sp_adj = sp_adj.to(device)
#    labels = labels.cuda()
#    idx_train = idx_train.cuda()
#    idx_val = idx_val.cuda()
#    idx_test = idx_test.cuda()


cnt_wait = 0
best = 1e9
best_t = 0



hold_best_model = model.state_dict()
for epoch in range(args.nb_epochs): #
    model.train()
    optimiser.zero_grad()
    res = model(features, adj_ori, args.negative_num, sp_adj, None, None) 
    loss = args.alpha*process.mi_loss_jsd(res[0], res[1]) + args.beta*process.mi_loss_jsd(res[2], res[3]) + args.gamma*process.reconstruct_loss(res[4], adj_target)
    #print('Epoch:', (epoch+1), '  Loss:', loss)

    if epoch%500==0:
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            hold_best_model = model.state_dict()
            print('Epoch:', (epoch+1), 'Best found.  Loss:', loss)
            torch.save(model.state_dict(), '/home/fatema/scratch/best_gmi'+'_exp1_C1'+'.pth.tar')
        else:
            print('Epoch:', (epoch+1))
            cnt_wait += 1

    if cnt_wait == args.epoch_flag:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t+1))
model.load_state_dict(hold_best_model)

#model.load_state_dict(torch.load('best_gmi.pkl'))

embeds = model.embed(features, sp_adj)
X_embedding = embeds.cpu().detach().numpy()

#with gzip.open('/home/fatema/scratch/best_gmi_X_embedding_exp1_C1', 'wb') as fp:
#    pickle.dump(X_embedding, fp) 
    
np.save('/home/fatema/scratch/best_gmi_X_embedding_exp1_C1', X_embedding)
print("GMI is finished")
    
########################################################################################################
train_embs = embeds[0, idx_train]

# val_embs = embeds[0, idx_val]      # typically, you could use the validation set
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

accs = []
xent = nn.CrossEntropyLoss()
iter_num = process.find_epoch(args.hid_units, nb_classes, train_embs, train_lbls, test_embs, test_lbls)
for _ in range(50): 
    log = LogReg(args.hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(iter_num):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print(acc * 100)
    accs.append(acc * 100)

accs = torch.stack(accs)
print('Average accuracy:', accs.mean())
print('STD:', accs.std())
Footer
© 2023 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Docs
    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

GMI/execute.py at master · zpeng27/GMI 
