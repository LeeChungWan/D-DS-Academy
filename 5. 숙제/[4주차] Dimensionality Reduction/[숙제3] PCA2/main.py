from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def plotting(X, filename):
    # plot 64 images
    fig=plt.figure(figsize=(6,6)) # setup a figure 6 inches by 6 inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8,8, i+1, xticks=[], yticks=[])
        ax.imshow(np.reshape(X[i], (64,64)), cmap=plt.cm.bone, interpolation='nearest')

    plt.savefig(filename)
    elice_utils.send_image(filename)

    return

def get_n_components(X, keep_ratio):
    # 몇개의 principal components를 사용할 것인지를 정하는 함수
    # (k개의 principal components를 통해 계산되는 variance)/(전체 variance)가 keep_ratio를 넘는 최소 k를 return 해 주세요.
    """
    k = 0
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / (np.std(X[:,i]) + (10**-10))
    cov = np.cov(X.T)
    """
    k=0
    cov = np.cov(X.T)
    u, s, vh = np.linalg.svd(cov)
    sorted_sigma = sorted(s, reverse = True)
    sum_sigma = 0
    while sum_sigma/sum(s) <= keep_ratio:
        sum_sigma += sorted_sigma[k]
        k += 1
    return k

def main():
    # Load data
    sample_idx = np.random.choice(list(range(400)), 64, replace=False)
    X=fetch_olivetti_faces().data[sample_idx,:] # (64, 4096)

    # Plot original data
    print('Original data')
    plotting(X, 'faces.png')

    # Get number of dimensions
    keep_ratio = 0.85
    k = get_n_components(X, keep_ratio)

    # do PCA with selected n_components
    pca_model = PCA(n_components=k)
    face_pca_array = pca_model.fit_transform(X)

    # Plot compressed data: Let's see how much of the variance is retained
    print('Reconstructed data with {} principal components'.format(k))
    plotting(pca_model.inverse_transform(face_pca_array), 'faces_compressed.png')

    return

if __name__ == '__main__':
    main()
    
