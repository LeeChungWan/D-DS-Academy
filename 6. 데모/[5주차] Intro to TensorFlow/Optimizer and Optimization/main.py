import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import elice_utils

# Create placeholder
a = tf.placeholder(dtype = tf.float64)
b = tf.placeholder(dtype = tf.float64)

# Create variable
x = tf.Variable(initial_value = 0., dtype = tf.float64, name = 'x')
y = tf.Variable(initial_value = 0., dtype = tf.float64, name = 'y')

# Create error function
f = tf.sqrt(tf.square(x - a) + tf.square(y - b))

# Create optimizer and minimization tensor
optimizer = tf.train.AdamOptimizer(learning_rate = 0.5)
opt_minimize = optimizer.minimize(f)

# Create initialization tensor
init_op = tf.global_variables_initializer()

# Create session
sess = tf.Session()

# Initialize variables
sess.run(init_op)

def proc():
    out = sess.run([f, x, y], feed_dict)
    outlist = [out]

    # Optimize variables x and y
    while(True):
        # Step forward using Adam gradient descent
        sess.run(opt_minimize, feed_dict)

        # Save trajectory
        out = sess.run([f, x, y], feed_dict)
        outlist.append(out)

        # Conditionally terminate optimization
        if out[0] < 0.1:
            break

	# Plot trajectories
    outlist = np.array(outlist)

    domain = np.linspace(-5., 5., num = 100)

    x1, x2 = np.meshgrid(domain, domain)
    heat = np.sqrt(np.square(x1 - feed_dict[a]) + np.square(x2 - feed_dict[b]))

    fig = plt.figure()
    CS = plt.contour(x1, x2, heat, cmap = 'rainbow')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(outlist[:, 1], outlist[:, 2], 'k-', marker = 'o')
    plt.scatter(outlist[0, 1], outlist[0, 2], marker = '^', color = 'red', s = 500)
    plt.scatter(outlist[-1, 1], outlist[-1, 2], marker = 'v', color = 'blue', s = 500)
    plt.savefig('res.png')
    elice_utils.send_image('res.png')

# Set center point (3., 3.) and optimize x and y
feed_dict = {a : 3., b : 3.}
proc()

# Set center point (-3., 3.) and optimize x and y
feed_dict = {a : -3., b : 3.}
proc()

# Set center point (-3., -3.) and optimize x and y
feed_dict = {a : -3., b : -3.}
proc()

# Set center point (3., -3.) and optimize x and y
feed_dict = {a : 3., b : -3.}
proc()
