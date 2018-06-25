import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC  # SVC stands for : Support Vector Classification
import elice_utils


# Functions linear_func1, linear_func2, and generate_data is used to generate random datapoints. You do not need to change this function
def nonlinear_func1(x):
    l = len(x)
    return (0.1 * pow(x, 3) + 0.2 * pow(x, 2) + 0.5 * x + 1000 + 3000 * np.random.random(l))

def nonlinear_func2(x):
    l = len(x)
    return (0.1 * pow(x, 3) + 0.2 * pow(x, 2) - 0.5 * x - 1000 - 3000 * np.random.random(l))

def generate_data(n):
    np.random.seed(32840091)

    x1_1 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_1 = nonlinear_func1(x1_1)
    x1_2 = (np.random.random(int(0.5 * n)) - 0.5) * 100
    x2_2 = nonlinear_func2(x1_2)
    y_1 = np.ones(int(0.5 * n))
    y_2 = -1 * np.ones(int(0.5 * n))

    x1 = np.concatenate((x1_1, x1_2))
    x2 = np.concatenate((x2_1, x2_2))
    y = np.concatenate((y_1, y_2))
    X = np.array(list(zip(x1, x2)))

    return (X, y)

def svm(X, y):
    clf = SVC(kernel='rbf', C=np.inf, gamma=1e-8)
    clf.fit(X, y)
    return (clf)

def draw(X, y, clf):
    filename = "nonlinear_svm.png"

    ###These are to make the figure clearer. You don't need to change this part.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.5  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='#CE5A57', alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c = ['#444C5C' if yy == 1 else '#78A5A3' for yy in y], edgecolor='none', s=30)

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()

if __name__ == '__main__':
    X, y = generate_data(100)
    clf = svm(X, y)
    draw(X, y, clf) 
