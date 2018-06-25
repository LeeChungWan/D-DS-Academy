#-*- coding: utf-8 -*-

import numpy as np

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import elice_utils

def generate_samples():
    "과제에서 사용할 데이터셋을 생성하는 함수입니다. 바꾸지 마세요!"
    mu1 = np.array([50, 35]) ; cov1 = np.array([[300, 100], [100, 40]])
    mu2 = np.array([45, 55]) ; cov2 = np.array([[300, 50], [50, 16]])

    np.random.seed(98749425)
    samples_1 = np.random.multivariate_normal(mu1, cov1, 100)
    samples_2 = np.random.multivariate_normal(mu2, cov2, 100)

    data = np.concatenate((samples_1, samples_2))
    labels = np.array([0]*100 + [1]*100)
    return data, labels

def plotting_samples(samples, color):
    "생성된 표본을 시각화합니다. 표본에 부여된 색상은 정답을 의미합니다."

    def deco_plot(ax):
        ax.set_xlabel("X1", fontsize=15); ax.set_ylabel("X2", fontsize=15)
        ax.set_ylim([0,100]); ax.set_xlim([0,100])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(samples[:,0], samples[:,1], c=color, s=20, alpha=.7, linewidth=0)
    deco_plot(ax)
    plt.savefig('samples.png')
    elice_utils.send_image('samples.png')
    plt.close()

def k_means_clustering(data):
    "Scikit-Learn 라이브러리를 이용해 K=2인 k-means clustering을 수행하고, 예측된 각 표본별 Cluster Label를 리턴하세요."
    "군집화함수의 인수중 하나, random_state=0으로 설정해주세요."
    "리턴값 예시: [0, 1, 1, 0, 1, 0, ...]"
    kmeans = kmeans = KMeans(n_clusters = 2, random_state = 0).fit(data)
    return kmeans.labels_

def GMM(data):
    "Scikit-Learn 라이브러리를 이용해 K=2인 GMM을 수행하고, 예측된 각 표본별 Cluster Label을 리턴하세요."
    "군집화함수의 인수 : random_state=0, tol=1e-10으로 설정해주세요. (grader 때문입니다.)"
    "리턴값 예시: [0, 1, 1, 0, 1, 0, ...]"
    GMM = GaussianMixture(n_components=2, tol=1e-10, random_state=0).fit(data)
    return GMM.predict(data)

def clustering(data):
    "k-means와 GMM중 더 나은 clustering algorithm을 골라서 Scikit-Learn 라이브러리를 이용해 학습하세요"
    "예측된 각 표본별 Cluster Label을 리턴하세요. 위에서 사용한 군집화함수에서 사용한 인수를 동일하게 적용해주세요. (grader 때문입니다.)"
    "리턴값 예시: [0, 1, 1, 0, 1, 0, ...]"
    GMM = GaussianMixture(n_components=2, tol=1e-10, random_state=0).fit(data)
    return GMM.predict(data)

def plot_labels(data, labels):
    "Clusering 결과를 시각화합니다. 인수로 데이터, 예측된 레이블을 받습니다."

    def deco_plot(ax):
        ax.set_xlabel("X1", fontsize=15); ax.set_ylabel("X2", fontsize=15)
        ax.set_ylim([0,100]); ax.set_xlim([0,100])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:,0], data[:,1], c=labels, s=20, alpha=.7, linewidth=0)
    deco_plot(ax)
    plt.savefig("results.png")
    elice_utils.send_image("results.png")
    plt.close()

def main():
    "표본 혹은 군집화 결과를 시각적으로 확인하고 싶다면 주석을 해제하세요."
    data, labels = generate_samples()
    #plotting_samples(data, labels)

    predicted_labels = k_means_clustering(data)
    #plot_labels(data, predicted_labels)

    predicted_labels = GMM(data)
    #plot_labels(data, predicted_labels)

    return

if __name__ == '__main__':
    main()
