#-*- coding: utf-8 -*-

import numpy as np

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
import elice_utils


def generate_samples():
    "과제에서 사용할 데이터셋을 생성하는 함수입니다. 바꾸지 마세요!"
    mu1 = np.array([20, 30]) ; cov1 = np.array([[70, 30], [30, 70]])
    mu2 = np.array([70, 30]) ; cov2 = np.array([[50, -30], [-30, 50]])
    mu3 = np.array([50, 70]) ; cov3 = np.array([[50, -20], [-20, 50]])

    np.random.seed(98749425)
    samples_1 = np.random.multivariate_normal(mu1, cov1, 100)
    samples_2 = np.random.multivariate_normal(mu2, cov2, 100)
    samples_3 = np.random.multivariate_normal(mu3, cov3, 100)

    data = np.concatenate((samples_1, samples_2, samples_3))
    labels = np.array([0]*100 + [1]*100 + [2]*100)
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


def GMM_find_k(data):
    "Model selection 파트입니다."
    "Scikit-Learn 라이브러리를 이용해 여러 가지 K (2~10)에 대해 GMM을 수행하고, aic를 이용해 적절한 K를 찾습니다."
    "군집화함수의 인수중 하나, random_state=0으로 설정해주세요. (grader 때문입니다.)"
    "Return : 주이진 데이터에 대해 최저의 AIC값을 갖는 number of components K"
    minimum_AIC = 10000000
    for i in range(2,11):
        GMM = GaussianMixture(n_components=i, random_state=0).fit(data)
        if GMM.aic(data) < minimum_AIC:
            minimum_AIC = GMM.aic(data)
            min_index = i
    return min_index


def GMM(data, k):
    GMM = GaussianMixture(n_components=k, random_state=0).fit(data)
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
    "표본 혹은 best_k가 주어졌을 때 군집화 결과를 시각적으로 확인하고 싶다면 주석을 해제하세요."
    data, labels = generate_samples()
    #plotting_samples(data, labels)

    best_k = GMM_find_k(data)
    #predicted_labels = GMM(data, best_k)
    #plot_labels(data, predicted_labels)

    return

if __name__ == '__main__':
    main()
