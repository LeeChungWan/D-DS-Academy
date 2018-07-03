import numpy as np
import pandas as pd

def user_factor_grad(data, theta, X, lam):
    # Implement Here
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        for i in range(len(data)):
            if np.isnan(data[i][j]) == False:
                g[j] += X[i] * (np.dot(theta[j].T, X[i]) - data[i][j])
        g[j] += lam * theta[j]
    return g

def item_factor_grad(data, theta, X, lam):
    # Implement Here
    g = np.zeros_like(X)
    for i in range(len(data)):
        for j in range(len(theta)):
            if np.isnan(data[i][j]) == False:
                g[i] += theta[j] * (np.dot(theta[j].T, X[i]) - data[i][j])
        g[i] += lam * X[i]
    return g

def error_function(data, theta, X, lam):
    # Initialize error function value
    f = 0

    # Error values
    f += 0.5 * np.nansum(np.square(data - np.matmul(X, theta.T)))

    # Ridge penalty values
    f += 0.5 * lam * (np.sum(np.square(theta)) + np.sum(np.square(X)))

    return f

def gradient_descent(data, theta0, X0, lam, alpha, max_iter, fun_tol):
    # Initialize parameters
    num_iter = 1
    diff = fun_tol + 1

    # Initialize search points
    theta = np.copy(theta0)
    X = np.copy(X0)

    prev_f = error_function(data, theta, X, lam)

    print('Run gradient descent algorithm')
    print('iter : %d, error : %.3f' % (num_iter, prev_f))

    while num_iter < max_iter and diff > fun_tol:
        # Print
        if num_iter % 20 == 0:
            print('iter : %d, error : %.3f' % (num_iter, prev_f))

        # Update user factor
        theta_grad = user_factor_grad(data, theta, X, lam)
        theta = theta - alpha * theta_grad

        # Update item factor
        X_grad = item_factor_grad(data, theta, X, lam)
        X = X - alpha * X_grad

        # Update error function value
        next_f = error_function(data, theta, X, lam)

        # Update parameters
        diff = abs(next_f - prev_f)
        prev_f = next_f
        num_iter += 1

    print('Terminated')
    return theta, X, next_f

def main():
    # Set up dataset
    data = np.array([[5, 5, 0, 0],
                     [5, np.nan, np.nan, 0],
                     [np.nan, 4, 0, np.nan],
                     [0, 0, 5, 4],
                     [0, 0, 5, np.nan]])

    user = ['NoSyu', 'Jin', 'Yeong', 'Bak']
    item = ['TRAIN TO BUSAN', 'Captain America: Civil War', 'THE WAILING', 'La La Land', 'Bridget Jones`s Baby']

    # Print dataset
    print('Print dataset')
    print(pd.DataFrame(data = data, index = item, columns = user))

    # Initialize user and item factors
    K = 3
    Nm = data.shape[0]
    Nu = data.shape[1]

    theta0 = np.random.normal(size = [Nu, K])
    X0 = np.random.normal(size = [Nm, K])

    # Initialize parameters
    lam = 0.5
    alpha = 1e-2
    max_iter = 1000
    fun_tol = 1e-3

    # Run gradient descent algorithm
    theta, X, next_f = gradient_descent(data, theta0, X0, lam, alpha, max_iter, fun_tol)

    # Print predictions of unrated items
    preds = np.matmul(X, theta.T)
    pd.options.display.float_format = '{:,.1f}'.format
    print(pd.DataFrame(data = preds, index = item, columns = user))

    print('Print predictions of unrated items')
    for i in range(Nm):
        for j in range(Nu):
            if np.isnan(data[i, j]):
                print('Predict rating of %s for %s is %.2f' % (item[i], user[j], preds[i, j]))

    return

if __name__ == '__main__':
    main()
