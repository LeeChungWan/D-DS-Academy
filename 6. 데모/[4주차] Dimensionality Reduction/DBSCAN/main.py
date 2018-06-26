from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import elice_utils

#### Generate data ####

X = []

with open("data.csv", 'r') as openfileobject:
    for __line in openfileobject:
        X.append([float(v) for v in __line.split()])

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
