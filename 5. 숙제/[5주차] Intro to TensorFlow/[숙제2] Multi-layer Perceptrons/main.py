from elice_utils import EliceUtils
elice_utils = EliceUtils()
import numpy as np
import tensorflow as tf

def get_feed_dict(w1, b1, w2, b2):
    feed_dict = {}
    feed_dict[w1] = [[3., 1., 0.], [-4., 4., 0.]]
    feed_dict[b1] = [[5/3., 7., -1]]
    feed_dict[w2] = [[0.], [0.], [0.]]
    feed_dict[b2] = [-0.4]
    return feed_dict

def main():

    ## build network
    ## we need input(data_in), w1, b1, w2, b2, and output(y)
    ## 문제를 풀기 위해 추가로 필요한 것(node, weight등)이 있다면 자유롭게 network를 만들어보세요.
    data_in = tf.placeholder(tf.float32, shape=[1, 2])
    w1 = tf.placeholder(tf.float32, shape=[2, 3])
    b1 = tf.placeholder(tf.float32, shape=[1, 3])
    L1 = tf.matmul(data_in, w1) + b1

    w2 = tf.placeholder(tf.float32, shape=[3, 1])
    b2 = tf.placeholder(tf.float32, shape=[1])
    y = tf.matmul(L1, w2) + b2
    ## calculate weight mannually for practice
    try:
        feed_dict = get_feed_dict(w1, b1, w2, b2)
        feed_dict[data_in] = [[0., 0.]]
    except:
        print('Build Network first!')
        return

    ## return the output
    with tf.Session() as sess:
        y = sess.run(y, feed_dict=feed_dict)
        if y > 0:
            y = 1
        else:
            y = 0
    print(feed_dict[data_in])
    print(y)

if __name__ == "__main__":
    main()
