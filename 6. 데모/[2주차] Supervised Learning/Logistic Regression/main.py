import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
import elice_utils


def main():
    data = pd.read_csv("./data/Default.csv", header=0)
    response_var = 1
    y_vec = data.ix[:, response_var].as_matrix()
    x_vec = data.ix[:, 3].as_matrix().reshape(-1, 1)

    one_var_default(x_vec, y_vec)


def one_var_default(x_vec, y_vec):
    filename = "default_logit_fig.png"

    regr_linear = linear_model.LinearRegression()
    regr_linear.fit(x_vec, y_vec)

    regr_logistic = linear_model.LogisticRegression()
    regr_logistic.fit(x_vec, y_vec)

    print("Independent variable: {}".format("Balance"))
    print("Coefficients: {}".format(regr_logistic.coef_))
    print("Intercept: {}".format(regr_logistic.intercept_))

    x_minmax = np.arange(x_vec.min(), x_vec.max()).reshape(-1, 1)
    plt.plot(x_vec, regr_linear.predict(x_vec), color='blue', linewidth=3)
    plt.plot(x_minmax, regr_logistic.predict_proba(x_minmax)[:, 1], color='red', linewidth=3)
    plt.scatter(x_vec, y_vec, color='black')

    plt.xlim((x_vec.min(), x_vec.max()))

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()


if __name__ == "__main__":
    main()
