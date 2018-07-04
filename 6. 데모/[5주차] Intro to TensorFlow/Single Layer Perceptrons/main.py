import tensorflow as tf

# Define True and False as floating type
T, F = 1., -1.

# Define the number of perceptrons
n_perceptron = 2

# Create placeholders
train_in = tf.placeholder(dtype = tf.float32, shape = [4, 2])
train_out = tf.placeholder(dtype = tf.float32, shape = [4, 1])

# Create perceptron parameters (Variables)
w1 = tf.Variable(tf.random_normal([2, n_perceptron]))
b1 = tf.Variable(tf.zeros([n_perceptron]))
w2 = tf.Variable(tf.random_normal([n_perceptron, 1]))
b2 = tf.Variable(tf.zeros([1]))

# First perceptron layer (Hyperbolic tangent gate)
out = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))

# Output perceptron
y = tf.tanh(tf.add(tf.matmul(out, w2), b2))

# Define error function
error = train_out - y
mse = tf.reduce_mean(tf.square(error))

# Create initialization tensor
init_op = tf.global_variables_initializer()

# Create optimization tensor
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# Create session
sess = tf.Session()

# Initialize variables
sess.run(init_op)

# Set up XOR problem
feed_dict = {
        train_in : [[T, T],
                    [T, F],
                    [F, T],
                    [F, F]],
        train_out : [[F],
                     [T],
                     [T],
                     [F]]
        }

# Print out the result of untrained perceptrons
print('Untrained perceptrons')
for ele in sess.run(y, feed_dict):
    if(ele > 0):
        print('T')
    else:
        print('F')

# Train single layer perceptrons network
for itr in range(5000):
    f, _ = sess.run([mse, train], feed_dict)

    if f < 0.01:
        print('Iter : %d, Trained mse value : %.4f' % (itr, f))
        break

# Print out the result of trained perceptrons
print('Trained perceptrons')
for ele in sess.run(y, feed_dict):
    if(ele > 0):
        print('T')
    else:
        print('F')
