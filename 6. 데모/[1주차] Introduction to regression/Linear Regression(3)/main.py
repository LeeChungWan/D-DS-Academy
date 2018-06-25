import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv("./data/Advertising.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[range(1, 4)])
    x_mat = data.ix[:, range(1, 4)].as_matrix().reshape(-1, 3)
    multi_var_advertising(x_mat, y_vec, x_label)


def multi_var_advertising(x_mat, y_vec, x_label, rs=108):
    x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2, random_state=rs)

    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    print("Independent variables: {}".format(x_label))
    print("Coefficients: {}".format(regr.coef_))
    print("Intercept: {}".format(regr.intercept_))
    print("RSS on test data: {}".format(np.sum((predicted_y_test - y_test) ** 2)))
    print("R^2 on test data: {}".format(r2_score(y_test, predicted_y_test)))
    print()


if __name__ == "__main__":
    main()
