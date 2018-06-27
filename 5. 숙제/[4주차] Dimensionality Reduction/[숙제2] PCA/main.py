import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import pylab as plt
import matplotlib.cm as cm
import elice_utils

def sample_data(n_samples):
    X = np.loadtxt("data/mnist2500_X.txt");
    labels = np.loadtxt("data/mnist2500_labels.txt");

    #sampling
    np.random.seed(0)
    sample_idx = np.random.choice(list(range(2500)), n_samples, replace=False)
    sampled_labels = labels[sample_idx]
    sampled_X = X[sample_idx]
    return sampled_X, sampled_labels


def plotting(Y, labels):
    # plot the results
    legend_ = []; colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in sorted(list(set(labels))):
        idxs = (labels==i).nonzero()
        l = plt.scatter(np.squeeze(Y[idxs,0]), Y[idxs,1], 20, color=colors[int(i)])
        legend_.append(l)
    plt.legend(legend_, list(range(10)), loc='center left', ncol=1, fontsize=8, bbox_to_anchor=(1, 0.5))
    plt.savefig("result.png");
    elice_utils.send_image("result.png")
    return


def pca(X, no_dims):
    # implement pca function here.
    # Don't use scikit-learn, please use numpy only.
    print(X.shape)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / (np.std(X[:,i]) + (10**-10))
    cov = np.cov(X.T)
    u, s, vh = np.linalg.svd(cov)
    principal_component_U = u[:,0]
    projection_Z = []
    for i in range(X.shape[0]):
        add_list = []
        for j in range(no_dims):
            add_list.append(np.dot(u[:,j].T, X[i].T))
        projection_Z.append(add_list)
    return projection_Z

def main():
    # load data
    X, labels = sample_data(1000)

    # run pca
    Y = pca(X, 2)

    # plotting
    # plotting(Y, labels)

    return sys.modules


if __name__ == '__main__':
    main()
