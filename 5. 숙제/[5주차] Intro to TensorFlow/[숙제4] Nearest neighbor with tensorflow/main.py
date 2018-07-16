import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def load_data():

    mnist = input_data.read_data_sets("./data/", one_hot=True)
    x_train, y_train = mnist.train.next_batch(5000) #5000 for training (nn candidates)
    x_test, y_test = mnist.test.next_batch(200) #200 for testing

    return x_train, y_train, x_test, y_test

def main():

    # Write the code here
    x_train, y_train, x_test, y_test = load_data()

    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])

    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        # loop over test data
        for i in range(len(x_test)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: x_train, xte: x_test[i, :]})
            # Calculate accuracy
            if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
                accuracy += 1./len(x_test)
    return accuracy

if __name__ == "__main__":
    main()
