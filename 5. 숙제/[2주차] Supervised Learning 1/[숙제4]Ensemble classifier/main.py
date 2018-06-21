import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import datasets

def Ensemble_clf(clfs, X, w):
    # Compute probabilities for targets from each classifier
    probs = [clf.predict_proba(X) for clf in clfs]
    ensemble = np.zeros(probs[0].shape)
    for i in range(len(probs)):
        ensemble += probs[i] * w[i]
    ensemble /= len(probs) * len(probs[1])
    result = []
    for i in range(len(ensemble)):
        result.append(np.argmax(ensemble[i]))
    return result

def main():
    '''
    Iris dataset에 대해 Classifier 1, 2, 3 및 Ensemble classifer의 5 cross-validation test score를 측정합니다.
    '''
    # Set random seed
    np.random.seed(123)

    # Prepare data
    iris = datasets.load_iris()
    X, y = iris.data[:, 1:3], iris.target

    # Randomly permute data
    random_idx = np.random.permutation(np.arange(len(y)))
    X = X[random_idx]
    y = y[random_idx]

    # Define clfs
    clf1 = LogisticRegression(C = 1)
    clf2 = LogisticRegression(C = 0.5)
    clf3 = GaussianNB()
    clfs = [clf1, clf2, clf3]

    # Define weights
    w = [1., 1., 1.]

    # 5 cross-validation
    kf = KFold(n_splits=5)

    # Define variable to save test scores
    result = []

    for train, test in kf.split(X):
        score = []

        for clf in clfs:
            clf.fit(X[train], y[train])
            clf_score = clf.score(X[test], y[test])
            score.append(clf_score)

        ensemble_clf_score = np.average(y[test] == Ensemble_clf(clfs, X[test], w))
        score.append(ensemble_clf_score)

        result.append(score)

    # Print test score
    print('clf1 score : %.2f\nclf2 score : %.2f\nclf3 score : %.2f\nensemble clf score : %.2f' \
          % tuple(np.average(np.asarray(result), axis = 0).tolist()))

if __name__ == "__main__":
    main()
