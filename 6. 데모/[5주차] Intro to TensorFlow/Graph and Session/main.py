# Import tensorflow library
import tensorflow as tf

# Create tensors
t1 = tf.constant([1., 2., 3., 4.])
t2 = tf.constant([5, 6, 7, 8])

# Reshape tensor
t3 = tf.reshape(t1, [2, 2])

# Cast tensor dtype
t1_int = tf.cast(t1, tf.int32)

# Create operation tensor
try:
    op = t1 + t2
except:
    op = t2 - t1_int

# Create session
sess = tf.Session()

# Print results
print('Print Tensor 1 : ' + str(sess.run(t1)))
print('Print a type of Tensor 1 : ' + str(t1.dtype))
print('Print a shape of Tensor 1 : ' + str(t1.shape))
print('Print Tensor 2 : ' + str(sess.run(t2)))
print('Print a type of Tensor 2 : ' + str(t2.dtype))
print('Print Tensor 3 : ' + str(sess.run(t3)))
print('Print a shape of Tensor 3 : ' + str(t3.shape))
print('Print Tensor op : ' + str(sess.run(op)))
