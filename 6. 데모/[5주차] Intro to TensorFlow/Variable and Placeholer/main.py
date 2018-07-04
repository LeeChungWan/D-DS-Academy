import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import elice_utils

# Create tensors (Variables, placeholder, operation)
A = tf.Variable(initial_value = 1., dtype = tf.float64, name = 'Amplitude')
f = tf.Variable(initial_value = 1., dtype = tf.float64, name = 'Frequency')
x = tf.placeholder(dtype = tf.float64, name = 'x')
sine = A * tf.sin(f * x)

# Create feed for placeholder
feed_dict = {x : np.pi * np.linspace(-1., 1.)}

# Create session
sess = tf.Session()

# Fail to run session (Uninitialized variables)
try:
    print(sess.run([A, f]))
except:
    print('Initialize the variables')

# Initialize variables
sess.run(tf.global_variables_initializer())

# Fail to run session (Unfeeded placeholder)
try:
    y = sess.run(sine)
except:
    print('Feed the placeholder')

# Plot sine graph
plt.figure()
plt.plot(np.pi * np.linspace(-1., 1.), sess.run(sine, feed_dict))
plt.title('Plot sine graph from -1. to 1. with amplitude ' + str(sess.run(A)) + ' frequency ' + str(sess.run(f)))
plt.savefig('fig1.png')
elice_utils.send_image('fig1.png')

# Modify amplitude (Re-assign variable value)
plt.figure()
sess.run(A.assign(2.))
plt.plot(np.pi * np.linspace(-1., 1.), sess.run(sine, feed_dict))
plt.title('Plot sine graph from -1. to 1. with amplitude ' + str(sess.run(A)) + ' frequency ' + str(sess.run(f)))
plt.savefig('fig2.png')
elice_utils.send_image('fig2.png')

# Modify frequency (Re-assign variable value)
plt.figure()
sess.run(f.assign(2.))
plt.plot(np.pi * np.linspace(-1., 1.), sess.run(sine, feed_dict))
plt.title('Plot sine graph from -1. to 1. with amplitude ' + str(sess.run(A)) + ' frequency ' + str(sess.run(f)))
plt.savefig('fig3.png')
elice_utils.send_image('fig3.png')
