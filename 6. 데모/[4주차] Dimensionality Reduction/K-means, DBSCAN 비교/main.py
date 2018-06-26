from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import elice_utils

def doKMeans(X) :
    X = np.array(X)
    labels = set([0, 1, 2])

    colours = ['red','green','blue']

    # Initialize K-means center points
    centers = [np.random.uniform(np.min(X[:, col]), np.max(X[:, col]), [3, 1]) for col in range(X.shape[1])]
    centers = np.concatenate(centers, axis = 1)

    preds = np.zeros(len(X))

    #### Plot results of K-means clustering ####

    for i in range(10):
        # Proceed the next iteration of K-means clustering
        kmeans = KMeans(n_clusters = 3, n_init = 1, init = centers, max_iter = 1, random_state = 0).fit(X)
        preds = kmeans.labels_
        centers = kmeans.cluster_centers_

    # Create an empty pyplot figure
    plt.figure()

    # Scatter plot for each label
    for label in labels:
        idx = (preds == label)
        plt.scatter(X[idx, 0], X[idx, 1], color = colours[label], marker = '.')

    # Plot K-means center points
    plt.scatter(centers[:, 0], centers[:, 1], marker = 'x', color = 'black')

    # Set title
    plt.title('KMeans iteration : ' + str(i))

    # Show the plot
    filename = 'kmeans' + str(i) + '.png'
    plt.savefig(filename)
    elice_utils.send_image(filename)

def doDBSCAN(X) :
    colours = ['red','green','blue']

    db = DBSCAN(eps=2, min_samples=4).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    #### Plot results of K-means clustering ####

    # Create an empty pyplot figure
    plt.figure()

    # Scatter plot for each label

    uniqueLabels = list(set(labels))

    for label in uniqueLabels :
        idx = (labels == label)

        for j in range(len(idx)) :
            if idx[j] == True :
                plt.scatter(X[j][0], X[j][1], color = colours[label], marker = '.')

    # Set title
    plt.title('DBSCAN')

    # Show the plot
    filename = 'DBSCAN.png'
    plt.savefig(filename)
    elice_utils.send_image(filename)



#### Generate data ####

X = []

with open("data.csv", 'r') as openfileobject:
    for __line in openfileobject:
        X.append([float(v) for v in __line.split()])


doKMeans(X)
doDBSCAN(X)
