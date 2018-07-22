from elice_utils import EliceUtils
elice_utils = EliceUtils()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# Hyperparameters for Learning
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Hyperparameters for NN Architecture
n_hidden_1 = 256 # 1번째 hidden layer의 Neurons
n_hidden_2 = 256 # 2번째 hidden layer의 Neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([784, 256])),
    'h2': tf.Variable(tf.random_normal([256, 256])),
    'out': tf.Variable(tf.random_normal([256, 10]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([10]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def main():

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    '''
    (Q1) Tensorflow에서 각각의 함수에 대한 문서를 찾고, 어떤 argument 또는 method를 사용해야 할 지 채워보세요.

    '''

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    '''
    (Q2) MLP(1)의 문제를 참고하여 session을 채워나가며, 필요한 연산을 채워넣어보세요.

    '''

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

            '''
            아래 부분은 수정하지 마세요

            '''

        print("Optimization Finished!")
        accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                          Y: mnist.test.labels})

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", accuracy)


if __name__ == "__main__":
    main()
