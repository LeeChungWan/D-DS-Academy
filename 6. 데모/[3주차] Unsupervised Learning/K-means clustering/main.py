from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import elice_utils

#### Generate data ####

# Set mean and standard deviation of nomral distribution for each label
mu = np.array([[1, 2], [1, -2], [4, 0]])
std = 0.7

# Set the number of samples
Num = 50

# Sample from the normal distributions
X = np.concatenate([np.random.normal(loc = mu, scale = std) for i in range(Num)])

labels = set([0, 1, 2])

colours = ['red','green','blue']

# Initialize K-means center points
centers = [np.random.uniform(np.min(X[:, col]), np.max(X[:, col]), [3, 1]) for col in range(X.shape[1])]
centers = np.concatenate(centers, axis = 1)

preds = np.zeros(len(X))

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
