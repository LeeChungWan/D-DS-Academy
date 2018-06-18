import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import svm

def linear_kernel(X, Y, w):
    '''
    1-D weight vector, w에 대해 linear kernel function을 작성하세요
    '''

    M = np.dot(X*w, (Y*w).T)
    return M

def main():
    '''
    이 부분은 수정하지 않으셔도 됩니다.
    '''

    # Prepare dataset
    data = load_breast_cancer()

    X = data.data
    y = data.target
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    print('Number of data : %d' % n_rows)
    print('Number of columns : %d' % n_cols)

    # Number of training and test data
    n_train = int(n_rows * 0.75)
    print('Number of training_data : %d' % n_train)

    # Randomly seperate training and test data
    idx = np.random.permutation(len(data.data))
    idx_train = idx[:n_train]
    idx_test = idx[-n_train:]

    # SVM using linear kernel function
    print('Using linear kernel function')

    # Train SVM with linear kernel
    clf_linear = svm.SVC(kernel = 'linear')
    clf_linear.fit(X[idx_train], y[idx_train])

    # Print training score
    print('training error : %.3f' % clf_linear.score(X[idx_train], y[idx_train]))

    # Print test score
    print('test error : %.3f' % clf_linear.score(X[idx_test], y[idx_test]))

    # SVM using custom kernel function
    print('Using custom kernel function')

    try:
        # Randomly pick kernel weights
        w = np.random.random(n_cols)
        my_kernel = (lambda X1, X2 : linear_kernel(X1, X2, w))

        # Train SVM with custom kernel
        clf_custom = svm.SVC(kernel = my_kernel)
        clf_custom.fit(X[idx_train], y[idx_train])

        # Print training score
        print('training error : %.3f' % clf_custom.score(X[idx_train], y[idx_train]))

        # Print test score
        print('test error : %.3f' % clf_custom.score(X[idx_test], y[idx_test]))

    except Exception as e:
        # Print Error
        print('Error : %s' % e)

if __name__ == "__main__":
    main()
