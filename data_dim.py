import pickle
import numpy as np

net_type = 'resnet'
datasets = ['cifar10', 'cifar100', 'svhn']
# layer_index = 4
datasets_len = [50000, 50000, 73257]

for i in range(len(datasets)):
    outf = './gmm_multiclass_output/' + net_type + '_' + datasets[i] + '/'
    print(outf)
    
    for layer_index in range(5):
        print("Processing Layer: ", layer_index)
        
        '''
        pkl_file_name = outf + 'activtions_in_' + str(layer_index) + '.pkl'
        
        data = []
        with open(pkl_file_name, 'rb') as fr:
            try:
                while True:
                    data.extend(pickle.load(fr))
            except EOFError:
                pass
        
        data = np.asarray(data)
        n, channel_n, w_h = data.shape
        print(data.shape)'''
        pkl_file_name = outf + 'activtions_in_' + str(layer_index) + '.pkl'
        data_batch = []
        with open(pkl_file_name, 'rb') as fr:
            data_batch.extend(pickle.load(fr))
        data_batch = np.asarray(data_batch)
        _, channel_n, w_h = data_batch.shape
        n = datasets_len[i]
        data = np.empty((n,channel_n*w_h))
        print(data.shape)
        ########################## Random Projections for dimensionality reduction
        
        
        from sklearn.random_projection import GaussianRandomProjection
        rng = np.random.RandomState(42)
        transformer = GaussianRandomProjection(n_components=channel_n, random_state=rng)
        transformer.fit(data.reshape(n,-1))
        file_name = outf + 'dim_transformer_' + str(layer_index) + '.pkl'
        pickle.dump(transformer, open(file_name, 'wb'))
        
        del(data)
        del(data_batch)














# ########################## PCA dimensionality reduction
# from sklearn.decomposition import PCA
#
# pca_full = PCA(n_components=channel_n*w_h)
# pca_full.fit(data.reshape(n,-1))
# variance_full = np.sum(pca_full.explained_variance_)
# print('Total variance of the data: ', variance_full)
#
# pca=PCA(n_components=channel_n)
# pca.fit(data.reshape(n,-1))
# variance_reduced = np.sum(pca.explained_variance_)
# print("PCA 512 dim variance: ", variance_reduced)
#
# ########################## Averaging for dimensionality reduction
# data_avg = np.mean(data, axis = -1)
# pca_avg = PCA(n_components = channel_n)
# pca_avg.fit(data_avg)
# variance_avg = np.sum(pca_avg.explained_variance_)
# print("Averaging based reduction variance: ", variance_avg)
#

########################## Random Projections for dimensionality reduction


# from sklearn.random_projection import GaussianRandomProjection
# rng = np.random.RandomState(42)
# transformer = GaussianRandomProjection(n_components=channel_n, random_state=rng)
# X_gau= transformer.fit_transform(data.reshape(n,-1))
# pca_gau = PCA(n_components=channel_n)
# pca_gau.fit(X_gau)
# variance_gau = np.sum(pca_gau.explained_variance_)
# print("Gaussian Random 512 dim variance: ", variance_gau)
#


# from sklearn.random_projection import SparseRandomProjection
# rng = np.random.RandomState(42)
# transformer_sparse = SparseRandomProjection(n_components=channel_n, random_state=rng)
# X_sparse = transformer_sparse.fit_transform(data.reshape(n,-1))
# pca_sparse = PCA(n_components=channel_n)
# pca_sparse.fit(X_sparse)
# variance_sparse = np.sum(pca_sparse.explained_variance_)
# print("Sparse Random 512 dim variance: ", variance_sparse)
#




