import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def main():
    training_data = pd.read_csv("./data/Hitters.csv", header=0)
    response_var = -1
    y_train = training_data.ix[:, response_var].squeeze()

    x_train = training_data.ix[:, 1:-1].as_matrix()
    x_train = x_train.reshape(-1, x_train.shape[1])

    test_data = pd.read_csv("./data/Hitters_Test.csv", header=0)
    y_test = test_data.ix[:, response_var].squeeze()

    x_test = test_data.ix[:, 1:-1].as_matrix()
    x_test = x_test.reshape(-1, x_test.shape[1])

    # Linear Regression
    rss, r2, mse = multi_var_hitter(x_train, x_test, y_train, y_test)
    print("Linear Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print()

    # Ridge Regression
    best_lambda_ridge, best_lambda_lasso = get_best_lambda_value_ridge_lasso(training_data)
    rss, r2, mse = multi_var_hitter_ridge(x_train, x_test, y_train, y_test, best_lambda_ridge)
    print("Ridge Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_ridge))
    print()

    # lasso
    rss, r2, mse = multi_var_hitter_lasso(x_train, x_test, y_train, y_test, best_lambda_lasso)
    print("lasso Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_lasso))
    print()


def get_best_lambda_value_ridge_lasso(data):
    """
    Implement Here
    The grader will call this function to get the lambda value,
    and run the functions with hidden test data.
    Do not write exact value on best_lambda_ridge and best_lambda_lasso.
    You should implement the function to find the best lambda value.
    """
    response_var = -1
    y_vec = data.ix[:, response_var].squeeze()
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])

    from sklearn.linear_model import RidgeCV, LassoCV

    ridgeregr = RidgeCV(cv=10, alphas=np.logspace(0, 100, 100))
    ridgeregr.fit(x_mat, y_vec)
    lassoregr = LassoCV(cv=10, n_alphas=100)
    lassoregr.fit(x_mat, y_vec)
    best_lambda_ridge = ridgeregr.alpha_
    best_lambda_lasso = lassoregr.alpha_

    return best_lambda_ridge, best_lambda_lasso


def multi_var_hitter(x_train, x_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    print("Coefficients: {}".format(regr.coef_))
    return rss, r2, mse


def multi_var_hitter_ridge(x_train, x_test, y_train, y_test, best_lambda):
    """
    Implement Here
    """
    regr = linear_model.Ridge(best_lambda)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    print("Coefficients: {}".format(regr.coef_))
    return rss, r2, mse


def multi_var_hitter_lasso(x_train, x_test, y_train, y_test, best_lambda):
    """
    Implement Here
    """
    regr = linear_model.Lasso(best_lambda)

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)

    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    print("Coefficients: {}".format(regr.coef_))
    return rss, r2, mse


if __name__ == "__main__":
    main()
