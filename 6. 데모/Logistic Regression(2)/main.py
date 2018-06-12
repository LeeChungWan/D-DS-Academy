from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv("./data/Default.csv", header=0)
    response_var = 1
    y_vec = data.ix[:, response_var].as_matrix()
    x_vec = data.ix[:, 3].as_matrix().reshape(-1, 1)

    one_var_default_pred(x_vec, y_vec)


def one_var_default_pred(x_vec, y_vec, rs=108):
    x_train, x_test, y_train, y_test = train_test_split(x_vec, y_vec, test_size=0.2, random_state=rs)

    regr_logistic = linear_model.LogisticRegression()

    regr_logistic.fit(x_train, y_train)

    score = regr_logistic.score(x_test, y_test)

    print("Independent variable: {}".format("Balance"))
    print("Coefficients: {}".format(regr_logistic.coef_))
    print("Intercept: {}".format(regr_logistic.intercept_))
    print("Accuracy: {}".format(score))


if __name__ == "__main__":
    main()
