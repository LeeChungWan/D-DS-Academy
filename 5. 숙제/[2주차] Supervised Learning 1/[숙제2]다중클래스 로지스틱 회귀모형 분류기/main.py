from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np


def load_data():
    return datasets.load_iris(return_X_y=True)


def binary_classification(X, binary_label):
	# input : X (features), shape:(number of samples, number of features)
	# input : binary_label (1 or 0), shape:(number of samples, )
	# output : probability to belong to that class, shape:(number of samples, )
    predicted_p = 0
    lrmodel = LogisticRegression()
    lrmodel.fit(X, binary_label)

    beta_0 = lrmodel.coef_[0]
    beta_1 = lrmodel.intercept_
    pridicted_p = lrmodel.predict_proba(X)[:][:,1]

    return pridicted_p

def multiclass_classification(X, y):
	# input : X (features), shape:(number of samples, number of features)
	# input : y (labels), shape:(number of samples,)
	# output : multiclass classification accuracy, shape:(1, )

    accuracy = 0
    label_0 = []
    label_1 = []
    label_2 = []
    for label in y:
        if label == 0:
            label_0.append(1)
            label_1.append(0)
            label_2.append(0)
        elif label == 1:
            label_0.append(0)
            label_1.append(1)
            label_2.append(0)
        elif label == 2:
            label_0.append(0)
            label_1.append(0)
            label_2.append(1)
    prob_0 = binary_classification(X, np.array(label_0))
    prob_1 = binary_classification(X, np.array(label_1))
    prob_2 = binary_classification(X, np.array(label_2))

    total_label = np.maximum(prob_0, prob_1)
    total_label = np.maximum(total_label, prob_2)
    count = 0
    for i in range(50):
        if total_label[0:50][i] == prob_0[0:50][i]:
            count += 1
        if total_label[50:100][i] == prob_1[50:100][i]:
            count += 1
        if total_label[100:150][i] == prob_2[100:150][i]:
            count += 1
    return count/len(y)

def main():
    data = load_data()
    result = multiclass_classification(data[0], data[1])
    return result



if __name__ == '__main__':
    main()
