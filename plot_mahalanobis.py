import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson, chi2, betaprime, lognorm
from sklearn.mixture import GaussianMixture
import pickle
import argparse


parser = argparse.ArgumentParser(description='Python code: Fit GMM to Mahalanobis Scores of In-Distribution datasets')
parser.add_argument('--net_type', default= 'resnet', help='resnet | densenet') ##  required=True,
args = parser.parse_args()
print(args)

# data_root = "./gen_output/resnet_cifar10"
# data_root = "./gen_output/"
data_root = "./gmm_multiclass_output/"

datasets = ['cifar10','cifar100','svhn']
# datasets = ['cifar10']
n_components = 1
class_conditional_gau_f = False

score_list = ['Mahalanobis_0.0']
score = score_list[0]
for dataset in datasets:
	outf = data_root + args.net_type + '_' + dataset + '/'
	file_name = os.path.join(outf, score+'_'+dataset+'.npy') ## 'Mahalanobis_0.0_cifar10.npy'
	label_file_name = os.path.join(outf, score+'_'+dataset+'_labels'+'.npy') ## 'Mahalanobis_0.0_cifar10_labels.npy'
	labels = np.load(label_file_name)
	labels = labels[:,0]
	print('file_name:',file_name)
	mahalanobis_scores = np.load(file_name)
	print(mahalanobis_scores.shape)
	num_layers= mahalanobis_scores.shape[1]
	N = mahalanobis_scores.shape[0]
	mahalanobis_scores = (-1)*mahalanobis_scores		## make score positive
	log_data = np.log(mahalanobis_scores)		## convert log-normal var to normal
	
	if class_conditional_gau_f == True:
		for i in range(100):
			class_conditional_scores = log_data[labels == i]
			class_conditional_scores = class_conditional_scores[:,i::100]
			print(class_conditional_scores.shape)
			gmm = GaussianMixture(n_components=n_components)
			gmm.fit(class_conditional_scores)
			print(gmm.score(class_conditional_scores))
			pickle.dump(gmm, open(outf+'gmm_'+str(n_components)+'_'+dataset+'class_'+str(i)+'.pkl', 'wb'))
			
	else:
		gmm = GaussianMixture(n_components=n_components, covariance_type = 'full')
		gmm.fit(log_data)
		log_ld = gmm.score(log_data)
		print((log_ld))
		pickle.dump(gmm, open(outf+'gmm_'+str(n_components)+'_'+dataset+'.pkl', 'wb'))


### sample from the gmm
#
# smpl_data, label = gmm.sample(n_samples=10000)
# smpl_data = np.exp(smpl_data)		## get log-norm values
# print(smpl_data.shape)
#
# num_dim = smpl_data.shape[1]
# N = smpl_data.shape[0]
#
#
# plot_root = './plots/resnet_cifar10/'
# if os.path.isdir(plot_root)==False:
# 	os.mkdir(plot_root)
# for i in range(num_dim):
# 	dim_data = smpl_data[:,i]
# 	plt.hist(dim_data, bins = N//20, histtype='stepfilled', density=True, color='r', label='Sampled Data', alpha =0.2)
# 	plt.hist(mahalanobis_scores[:,i], bins=N//20, histtype='stepfilled', density=True, color='b',label = 'Activation Data', alpha = 0.2)
# 	plt.xlabel('Mahalanobis Scores')
# 	plt.ylabel('Density')
# 	plt.legend()
# 	plt.title('Layer:'+str(i))
# 	plot_name = 'Layer_' + str(i)
# 	# plt.show()
# 	plt.savefig(plot_root+plot_name+'.png')
# 	plt.clf()
#
#
#
# ## load and see out of distribution data
# out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
# file_name = os.path.join(data_root,'Mahalanobis_0.0_cifar10_'+out_dist_list[0]+'.npy')
# print('file_name:',file_name)
# mahalanobis_scores = np.load(file_name)
# print(mahalanobis_scores.shape)
# num_layers= mahalanobis_scores.shape[1]
# N = mahalanobis_scores.shape[0]
#












# ### 1-D curve fit for log-normal
# density = []
# bins = []
# for i in range(num_layers):
# 	layer_data = mahalanobis_scores[:,i]
# 	d, b, _ = plt.hist(layer_data, bins= 100, histtype= 'bar',density=True)
# 	density.append(d)
# 	bins.append(b)
#
# plt.show()
# density = np.asarray(density)
# bins = np.asarray(bins)
# print(density.shape)
# print (bins.shape)
#
# bins_1 = bins[:,:-1]
#
#
# def log_norm_fit(X, s, loc, scale ):
# 	return lognorm.pdf(X, s, loc, scale)
#
#
# popt, pcov = curve_fit(log_norm_fit, bins_1[0,:], density[0,:])
#
# print(popt)
# print(pcov)











## histogram views
# plot_root = './plots/resnet_cifar10/'
# if os.path.isdir(plot_root)==False:
# 	os.mkdir(plot_root)
# all_flag = 1
# for i in range(num_layers):
# 	layer_data= mahalanobis_scores[:,i]
# 	plt.hist(layer_data, bins = N//100, histtype='stepfilled', label='Layer: '+str(i))
# 	plt.xlabel('Bins')
# 	plt.ylabel('Mahalanobis Scores')
# 	# plot_name = 'Layer_'+str(i)+'_Mahalanobis_Score_Histogram'
# 	# plt.title(plot_name)
# 	plt.legend()

# plt.savefig(plot_root+'all_layers_hist.png')
# plt.clf()


# d_centroid = []
# for i in range(len(d) - 1):
# 	d_centroid.append((d[i + 1] + d[i]) / 2)
# d_centroid = np.array(d_centroid)
#
#
# def get_fit_func(dist, nr_params):
# 	def fit_function_one(k, lamb1):
# 		if dist == poisson:
# 			return dist.pmf(np.round(k), lamb1)
# 		else:
# 			return dist.pdf(k, lamb1)
#
# 	def fit_function_two(k, lamb1, lamb2):
# 		return dist.pdf(k, lamb1, lamb2)
#
# 	if nr_params == 1:
# 		return fit_function_one
# 	elif nr_params == 2:
# 		return fit_function_two
#
#
# distributions = [poisson, chi2, betaprime]
# fit_funcs = [get_fit_func(poisson, 1), get_fit_func(chi2, 1), get_fit_func(betaprime, 2)]
# init_params = [20, 10, [10, 2]]
#
# best_ll = -1 * np.infty
# best_func = None
# for i in range(len(distributions)):
#
# 	parameters, cov_matrix = curve_fit(fit_funcs[i], d_centroid, p, p0=init_params[i])
#
# 	# average log likelihood
# 	ll = np.log(fit_funcs[i](dps_pred, *parameters))
# 	print(distributions[i].__class__.__name__, ':', ll.mean())
#
# 	if ll.mean() > best_ll:
# 		best_dist = distributions[i](*parameters)
# 		best_func = lambda x: fit_funcs[i](x, *parameters)
#
# 	x_plot = np.arange(d.min(), d.max())
# 	plt.figure()
# 	plt.plot(
# 			x_plot,
# 			fit_funcs[i](x_plot, *parameters),  # parameters
# 			linestyle='-',
# 			label='Fit result',
# 	)
# 	plt.plot(d_centroid, p, c='r')
# 	plt.show()


#
# def func(X, mean, sd ):
# 	Y = np.log(X) 		## assuming X is log-normal, Y should be multi-variate normal
#
