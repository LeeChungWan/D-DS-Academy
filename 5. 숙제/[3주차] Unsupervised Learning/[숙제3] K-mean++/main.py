import elice_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def sample(X, probs):
    idx_list = np.arange(len(X))
    random_idx = np.random.choice(
        idx_list,
        p=probs
    )

    return [X[random_idx]]
def compute_probs(X, centers):
    distances = []

    for i in range(len(X)):
        compare_dist = 10000
        for j in range(len(centers)):
            dist = np.sqrt(np.sum((X[i] - centers[j]) ** 2))
            compare_dist = min(compare_dist, dist)
        distances.append(compare_dist**2)
    distances /= sum(distances)
    return distances

def initialize(X, n_cluster):
    probs = np.ones(len(X)) / len(X)
    centers = sample(X, probs)
    while(len(centers) < n_cluster):
        probs = compute_probs(X, centers)
        next_x = sample(X, probs)
        centers = np.append(centers, next_x, axis = 0)
    return centers

def main():
	#### Generate data ####

    # Set mean and standard deviation of nomral distribution for each label
    mu = np.array([[1, 2], [1, -3], [5, 0]])
    std = 0.7

    # Set the number of samples
    Num = 50

    # Sample from the normal distributions
    X = np.concatenate([np.random.normal(loc = mu, scale = std) for i in range(Num)])
    y = np.concatenate([[0, 1, 2] for i in range(Num)])

    labels = set(y)

    colours = ['red','green','blue']

    # Initialize K-means center points
    centers = initialize(X, 3)

    preds = y

    #### Plot results of K-means clustering ####

    for i in range(10):
        # Create an empty pyplot figure
        plt.figure()

        # Scatter plot for each label
        for label in labels:
            idx = (preds == label)
            plt.scatter(X[idx, 0], X[idx, 1], color = colours[label], marker = '.')

        # Plot K-means center points
        plt.scatter(centers[:, 0], centers[:, 1], marker = 'x', color = 'black')

        # Set title
        plt.title('Iteration : ' + str(i))

        # Show the plot
        filename = 'kmeans' + str(i) + '.png'
        plt.savefig(filename)
        elice_utils.send_image(filename)

        # Proceed the next iteration of K-means clustering
        kmeans = KMeans(n_clusters = 3, n_init = 1, init = centers, max_iter = 1, random_state = 0).fit(X)
        preds = kmeans.labels_
        centers = kmeans.cluster_centers_

if __name__ == "__main__":
    main()
