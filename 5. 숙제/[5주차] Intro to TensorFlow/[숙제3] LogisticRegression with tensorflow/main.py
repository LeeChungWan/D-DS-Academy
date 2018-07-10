from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import elice_utils
import sys

def read_data():
    iris = pd.read_csv("data/Iris.csv")[:100]
    iris.Species = (iris.Species == 'Iris-versicolor').astype(int)
    x = iris.drop(labels=['Id', 'Species'], axis=1).values
    y = iris.Species.values
    return normalize(x), y

def normalize(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    return np.divide(data-data_min, data_max-data_min)

def logistic_regression(x_train, x_test, y_train, y_test):
    train_accuracy = 0
    test_accuracy = 0
    X = tf.placeholder(tf.float32, shape=[None, 4])
    Y = tf.placeholder(tf.float32, shape=[None])
    W = tf.Variable(tf.random_normal([4,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(1000):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})

        train_accuracy = sess.run(accuracy, feed_dict={X:x_train, Y:y_train})
        test_accuracy = sess.run(accuracy, feed_dict={X:x_test, Y:y_test})
    return train_accuracy, test_accuracy


def main():
    x, y = read_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train_accuracy, test_accuracy = logistic_regression(x_train, x_test, y_train, y_test)

    return sys.modules


if __name__ == '__main__':
    main()
