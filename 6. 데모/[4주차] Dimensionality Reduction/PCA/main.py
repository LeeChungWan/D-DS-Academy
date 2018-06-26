import sklearn.preprocessing
import numpy as np
import pandas as pd
import elice_utils
import scipy.spatial.distance
import operator
from scipy import linalg
from sklearn.decomposition import PCA

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def main():
    champs_df = pd.read_pickle('./data/champ_df.pd')
    champ_pca_array = run_PCA(champs_df, 2)

    plot_champions(champs_df, champ_pca_array)
    print(get_closest_champions(champs_df, champ_pca_array, "Ashe", 10))

def run_PCA(champs_df, num_components):
    # Scale Attributes
    scaler = sklearn.preprocessing.MinMaxScaler()
    champs = scaler.fit_transform(champs_df)

    # Run PCA
    pca_model = PCA(n_components=2)
    champ_pca_array = pca_model.fit_transform(champs)
    return champ_pca_array

def get_closest_champions(champs_df, champ_pca_array, champ_name, num_champions):
    # Get the champion index
    champ_list = champs_df.index.tolist()
    try:
        champ_idx = champ_list.index(champ_name)
    except:
        return "%s is not in the champion list" % champ_name

    # Get the euclidean distance
    # Use scipy.spatial.distance.euclidean(A, B)
    distance_from_current_champ = {}
    for i in range(0, len(champ_list)):
        distance_from_current_champ[champ_list[i]] = \
            scipy.spatial.distance.euclidean(
                champ_pca_array[champ_idx],
                champ_pca_array[i]
            )

    # Sort dictionary according to the value
    sorted_champs = sorted(distance_from_current_champ.items(), key = operator.itemgetter(1))

    # Return top 10 champs except current one
    if num_champions > len(champ_list) - 1:
        return "num_champions is too large"
    else:
        return sorted_champs[1:1+num_champions]

def plot_champions(champs_df, champ_pca_array):
    champ_names = champs_df.index.values

    x = champ_pca_array[:, 0]
    y = champ_pca_array[:, 1]
    difficulty = champs_df['difficulty'].values
    magic = champs_df['attack'].values

    plt.figure(figsize=(20, 10))

    plt.scatter(x, y,  c = magic, s = difficulty*1500, cmap = plt.get_cmap('Spectral'))

    for champ_name, x, y in zip(champ_names, x, y):
        plt.annotate(
            champ_name,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.savefig('image.png')
    elice_utils.send_image('image.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    main()
