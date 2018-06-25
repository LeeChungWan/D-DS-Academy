from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv("./data/Default.csv", header=0)
    response_var = 1
    y_vec = data.ix[:, response_var].as_matrix()
    x_mat = data.ix[:, range(2, 5)].as_matrix().reshape(-1, 3)

    multi_var_default(x_mat, y_vec)


def multi_var_default(x_mat, y_vec, rs=108):
    x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2, random_state=rs)

    regr_logistic = linear_model.LogisticRegression()

    regr_logistic.fit(x_train, y_train)

    score = regr_logistic.score(x_test, y_test)


    print("Indepedent variables: {}".format("ALL"))
    print("Coefficients: {}".format(regr_logistic.coef_))
    print("Intercept: {}".format(regr_logistic.intercept_))
    print("Accuracy: {}".format(score))


if __name__ == "__main__":
    main()
