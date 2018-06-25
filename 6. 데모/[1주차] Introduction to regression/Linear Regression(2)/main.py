import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import elice_utils


def main():
    data = pd.read_csv("./data/Advertising.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    for independent_var in range(1, 4):
        try:
            x_vec = data.ix[:, independent_var].as_matrix().reshape(-1, 1)
            x_label = data.columns[independent_var]

            one_var_advertising_pred(x_vec, y_vec, x_label, y_label)
        except ValueError:
            pass


def one_var_advertising_pred(x_vec, y_vec, x_label, y_label, rs=108):
    filename = "advertising_fig_simple_train_test_{}.png".format(x_label)

    x_train, x_test, y_train, y_test = train_test_split(x_vec, y_vec, test_size=0.2, random_state=rs)

    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    print(filename)
    print("Independent variable: {}".format(x_label))
    print("Coefficients: {}".format(regr.coef_))
    print("Intercept: {}".format(regr.intercept_))
    print("RSS on test data: {}".format(np.sum((predicted_y_test - y_test) ** 2)))
    print("MSE on test data: {}".format(mean_squared_error(y_test, predicted_y_test)))
    print("R^2 on test data: {}".format(r2_score(y_test, predicted_y_test)))
    # Another way to compute the R^2 score
    # print("R^2 on test data: {}".format(regr.score(x_test, y_test)))
    print()

    plt.scatter(x_train, y_train, color='black')
    plt.plot(x_train, regr.predict(x_train), color='blue', linewidth=3)

    plt.xlim((0, int(max(x_vec) * 1.1)))
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()


if __name__ == "__main__":
    main()
