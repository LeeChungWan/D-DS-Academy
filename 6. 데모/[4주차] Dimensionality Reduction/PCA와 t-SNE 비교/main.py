import matplotlib
matplotlib.use('Agg')

import numpy as Math
import pylab as Plot
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import sklearn.preprocessing

import elice_utils
import tsne

def sample_data(n_samples):
	X = Math.loadtxt("data/mnist2500_X.txt");
	labels = Math.loadtxt("data/mnist2500_labels.txt");

	#sampling
	Math.random.seed(0)
	sample_idx = Math.random.choice(list(range(2500)), n_samples, replace=False)
	sampled_labels = labels[sample_idx]
	sampled_X = X[sample_idx]
	return sampled_X, sampled_labels

def pca(X, no_dims):
	scaler = sklearn.preprocessing.MinMaxScaler()
	X_scaled = scaler.fit_transform(Math.array(X).astype('float64'))
	mu = Math.mean(X_scaled, axis=0)
	X_scaled -= mu

	model = PCA(n_components=no_dims)
	Y = model.fit_transform(X_scaled)
	return Y


def main():
	# load data
	print("Loading data...")
	X, labels = sample_data(1000)

	# run pca
	#print("Run Y = pca(X, no_dims) to perform PCA on your dataset.")
	#Y = pca(X, 2)

	# run tsne
	print("Run Y = tsne.tsne(X, no_dims, initial_dims, perplexity) to perform t-SNE on your dataset.")
	Y = tsne.tsne(X, 2, 50, 30.0);

	# plot the results
	legend_ = []; colors = cm.rainbow(Math.linspace(0, 1, 10))
	for i in sorted(list(set(labels))):
		idxs = (labels==i).nonzero()
		l = Plot.scatter(Math.squeeze(Y[idxs,0]), Y[idxs,1], 20, color=colors[int(i)])
		legend_.append(l)
	Plot.legend(legend_, list(range(10)), loc='center left', ncol=1, fontsize=8, bbox_to_anchor=(1, 0.5))
	Plot.savefig("result.png");
	elice_utils.send_image("result.png")
	return

if __name__ == "__main__":
	main()
