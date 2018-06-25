import numpy as np

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import chi2

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import elice_utils

def read_input():
    mus = [int(x) for x in input().split()]
    cov1 = [int(x) for x in input().split()]
    cov2 = [int(x) for x in input().split()]
    return mus, [cov1, cov2]

def generate_samples(mus, cov):
    return np.random.multivariate_normal(mus, cov, 1000)

def plotting_samples(ax, samples, mus, cov):
    ax.scatter(samples[:,0], samples[:,1], color='gray', s=.1, alpha=1)

def plot_cov_ellipse(ax, cov, pos, volume=.5, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)
    return ax

def deco_plot(ax):
    ax.set_xlabel("X1", fontsize=15)
    ax.set_ylabel("X2", fontsize=15)
    ax.set_ylim([0,100])
    ax.set_xlim([0,100])
    return

def MVN_2d_plot(ax, samples, mus, cov):
    plotting_samples(ax, samples, mus, cov)
    plot_cov_ellipse(ax, cov, mus, volume=.1, fc='red', a=.5, lw=0)
    plot_cov_ellipse(ax, cov, mus, volume=.3, fc='red', a=.3, lw=0)
    plot_cov_ellipse(ax, cov, mus, volume=.5, fc='blue', a=.2, lw=0)
    plot_cov_ellipse(ax, cov, mus, volume=.7, fc='blue', a=.1, lw=0)
    plot_cov_ellipse(ax, cov, mus, volume=.9, fc='blue', a=.05, lw=0)

def MVN_3d_plot(ax, mu1, mu2, cov1, cov2):
    X1 = np.arange(mu1[0] - 3*np.sqrt(cov1[0, 0]), mu1[0] + 3*np.sqrt(cov1[0, 0]), 6*np.sqrt(cov1[0, 0])/100)
    Y1 = np.arange(mu1[1] - 3*np.sqrt(cov1[1, 1]), mu1[1] + 3*np.sqrt(cov1[1, 1]), 6*np.sqrt(cov1[1, 1])/100)
    X2 = np.arange(mu2[0] - 3*np.sqrt(cov2[0, 0]), mu2[0] + 3*np.sqrt(cov2[0, 0]), 6*np.sqrt(cov2[0, 0])/100)
    Y2 = np.arange(mu2[1] - 3*np.sqrt(cov2[1, 1]), mu2[1] + 3*np.sqrt(cov2[1, 1]), 6*np.sqrt(cov2[1, 1])/100)


    X1_m, Y1_m = np.meshgrid(X1, Y1)
    X2_m, Y2_m = np.meshgrid(X2, Y2)
    X = np.append(X1_m, X2_m)
    Y = np.append(Y1_m, Y2_m)

    Z1 = multivariate_normal.pdf(np.array([X1_m,Y1_m]).T, mean=mu1, cov=cov1)
    Z2 = multivariate_normal.pdf(np.array([X2_m,Y2_m]).T, mean=mu2, cov=cov2)

    X = np.concatenate((X1_m, X2_m))
    Y = np.concatenate((Y1_m, Y2_m))
    Z = np.concatenate((Z1, Z2))

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax.set_zlabel("P(X)", fontsize=15)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter(''))

def main(mu1, mu2, cov1, cov2):
    samples_1 = generate_samples(mu1, cov1)
    samples_2 = generate_samples(mu2, cov2)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    MVN_2d_plot(ax1, samples_1, mu1, cov1)
    MVN_2d_plot(ax1, samples_2, mu2, cov2)
    deco_plot(ax1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    MVN_3d_plot(ax2, mu1, mu2, cov1, cov2)
    deco_plot(ax2)

    plt.savefig('test.png')
    elice_utils.send_image('test.png')
    return

if __name__ == '__main__':
    # Latent MVN distribution #1
    mu1 = np.array([40, 40])
    cov1 = np.array([[150, 30],
                    [30, 150]])

    # Latent MVN distribution #2
    mu2 =  np.array([60, 60])
    cov2 = np.array([[150, -25],
                    [-25, 150]])

    # Visualization
    main(mu1, mu2, cov1, cov2)
