import elice_utils
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.examples.tutorials.mnist import input_data

def load_data():

    mnist = input_data.read_data_sets("./data/", one_hot=True)
    x_train, y_train = mnist.train.next_batch(5000) #5000 for training (nn candidates)
    x_test, y_test = mnist.test.next_batch(200) #200 for testing

    return x_train, y_train, x_test, y_test

def train(x_train, y_train):

    # Parameters
    num_steps = 50 # Total steps to train
    batch_size = 1024 # The number of samples per batch
    k = 25 # The number of clusters
    num_classes = 10 # The 10 digits
    num_features = 784 # Each image is 28x28 pixels

    # Put placeholders for input data here
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    # Initialize K-means parameters here (KMeans)
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
    # Build K-means graph (kmeans.training_graph())
    training_graph = kmeans.training_graph()
    # You will need the output of kmeans.training_graph() to calculate the average distance
    if len(training_graph) > 6:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)
    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()
    # Start TensorFlow session and run the initializer
    sess = tf.Session()

    sess.run(init_vars, feed_dict={X: x_train})
    sess.run(init_op, feed_dict={X: x_train})
    # Training
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: x_train})
        if i % 10 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))
    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'idx')
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += y_train[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # return a model, sess and both placeholders to test the model

    return sess, accuracy_op, X, Y


def test(sess, accuracy_op, X, Y, x_test, y_test):

    accuracy = sess.run(accuracy_op, feed_dict={X: x_test, Y: y_test})
    print("Test Accuracy:", accuracy)

    return accuracy

def main():

    x_train, y_train, x_test, y_test = load_data()
    sess, accuracy_op, X, Y = train(x_train, y_train)
    accuracy = test(sess, accuracy_op, X, Y, x_test, y_test)

    return accuracy

if __name__ == "__main__":
    main()
