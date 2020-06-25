import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson, chi2, betaprime, lognorm
from sklearn.mixture import GaussianMixture
import pickle
import argparse
from sklearn.random_projection import GaussianRandomProjection

parser = argparse.ArgumentParser(description='Python code: Plot distribution of activations')
parser.add_argument('--net_type', default= 'resnet', help='resnet | densenet') ##  required=True,
args = parser.parse_args()
print(args)
data_root = './gmm_multiclass_output/'
# datasets = ['cifar10','cifar100','svhn']
datasets = ['cifar10']
datasets_len = [50000, 50000, 73257]
n_components = 1
avg_flag = False			## to do dimensionality reduction by averaging across channel HxW

score_list = ['Mahalanobis_0.0']
score = score_list[0]
for i in range(len(datasets)):
	outf = data_root + args.net_type + '_' + datasets[i] + '/'
	print(outf)
	for layer_index in range(5):
		print("Processing Layer: ", layer_index)
		pkl_file_name = outf + 'activtions_in_' + str(layer_index) + '.pkl'
		data = []
		first_batch_flag = 1
		with open(pkl_file_name, 'rb') as fr:
			try:
				while True:
					# data.extend(pickle.load(fr))
					temp_data = pickle.load(fr)
					temp_data = np.asarray(temp_data)
					_, channel_n, w_h = temp_data.shape

					if avg_flag == True:
						temp_data = np.mean(temp_data, axis= 2)
					else:
						if first_batch_flag == 1:
							transformer = GaussianRandomProjection(n_components=channel_n)
							n = datasets_len[i]
							dummy = np.empty((n, channel_n * w_h))
							transformer.fit(dummy)
							first_batch_flag = 0
							del(dummy)
						
						temp_data = transformer.transform(temp_data.reshape(temp_data.shape[0],-1))
					data.extend(temp_data)
					del(temp_data)
			except EOFError:
				pass

		data = np.asarray(data)
		n, channel_n= data.shape
		print(data.shape)
		
		for channel in range(20):
			channel_act = data[:, channel]
			plt.hist(channel_act, bins=n // 200, histtype='stepfilled', color='b', alpha=0.2)
			plt.show()
		
