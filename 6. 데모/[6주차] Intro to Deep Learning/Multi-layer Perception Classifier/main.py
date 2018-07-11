from sklearn.neural_network import MLPClassifier
import numpy as np
import elice_utils
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def main():
    # 1
    X, Y = read_data('case_0.txt') # try to use different datasets

    clf = train_MLP_classifier(X, Y)
    report_clf_stats(clf, X, Y)
    visualize(clf, X, Y)

def train_MLP_classifier(X, Y):
    # 2
    clf = MLPClassifier(hidden_layer_sizes=(5)) # try changing the number of hidden layers

    clf.fit(X, Y)

    return clf

def report_clf_stats(clf, X, Y):
    # 1. measure accuracy
    hit = 0
    miss = 0

    for x, y in zip(X, Y):
        if clf.predict([x])[0] == y:
            hit += 1
        else:
            miss += 1

    print("Accuracy: %.1lf%%" % float(100 * hit / (hit + miss)))

def read_data(filename):
    X = []
    Y = []

    with open(filename) as fp:
        N, M = fp.readline().split()
        N = int(N)
        M = int(M)

        for i in range(N):
            line = fp.readline().split()
            for j in range(M):
                X.append([i, j])
                Y.append(int(line[j]))

    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def visualize(clf, X, Y):
    X_ = []
    Y_ = []
    colors = []
    shapes = []

    plt.figure(figsize=(6, 6))

    for x, y in zip(X, Y):
        X_.append(x[1])
        Y_.append(x[0])
        if y == 0:
            colors.append('b')
        else:
            colors.append('r')

        if clf.predict([x])[0] == y:
            shapes.append('o')
        else:
            shapes.append('x')

    for x, y in zip(X, Y):
        c = '#87CEFA'
        if clf.predict([x])[0] == 1:
            c = '#fab387'
        plt.scatter(x[1], x[0], marker='s', c=c, s=1200, edgecolors='none')

    for _s, c, _x, _y in zip(shapes, colors, X_, Y_):
        plt.scatter(_x, _y, marker=_s, c=c, s=200)
    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == "__main__":
    main()
