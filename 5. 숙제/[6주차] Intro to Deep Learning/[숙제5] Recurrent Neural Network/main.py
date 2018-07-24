import numpy as np
import tensorflow as tf

# load data
def read_data():
    train_X = np.load("./imdb/train_X.npy")
    train_Y = np.load("./imdb/train_Y.npy")
    train_Y = np.array(train_Y, dtype=np.float32).reshape((-1, 1))
    test_X = np.load("./imdb/test_X.npy")
    test_Y = np.load("./imdb/test_Y.npy")
    test_Y = np.array(test_Y, dtype=np.float32).reshape((-1, 1))
    return train_X, train_Y, test_X, test_Y


# add padding to input sequences & count sequence lengths
def preprocsesing(input_sequence):
    seq_lens = np.zeros(len(input_sequence), dtype=np.int64)
    max_len = 100 # max sequence lengths is set to 100
    input_sequence_padded = 2 * np.ones((len(input_sequence), 100), dtype=np.int64) # out of vocab:2
    for i, seq in enumerate(input_sequence):
        input_sequence_padded[i,:len(seq)] = seq
        seq_lens[i] = len(seq)
    return input_sequence_padded, seq_lens


# build forward path of RNN classifier
def forward_path(train_X, train_X_lengths, test_X, test_X_lengths, hyperparams):
    #hyperparameters
    embedding_dims = hyperparams['embedding_dims']
    max_seq_lens = hyperparams['max_seq_len']
    n_of_unique_words = hyperparams['n_of_unique_words']
    hidden_size = hyperparams['n_of_hidden_units']

    #init word vectors
    init_width = 1/embedding_dims
    word_embedding_matrix = tf.Variable(
        tf.random_uniform([n_of_unique_words, embedding_dims], -init_width, init_width, dtype=tf.float32),
        name="embeddings",
        dtype=tf.float32)

    #inputs (placeholders)
    batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
    input_sentence_idx = tf.placeholder(tf.int64, shape=(None, max_seq_lens), name="input_sentence_placeholders")
    input_sentence_len = tf.placeholder(tf.int64, shape=(None), name="input_sentence_len")
    input_labels = tf.placeholder(tf.float32, shape=(None, 1), name="input_labels")
    placeholders = [input_sentence_idx, input_sentence_len, input_labels, batch_size]

    #convert input idx to embeddings

    input_sentence_emb = tf.nn.embedding_lookup(word_embedding_matrix, input_sentence_idx)

     ############### write code for 1 RNN layer (GRU) + 1 dense layer ###############

    # (1) RNN
    cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, input_sentence_emb, sequence_length=input_sentence_len, initial_state=initial_state, dtype=tf.float32)

    # (2) dense layer
    logits = tf.contrib.layers.fully_connected(_states[:], 1)
    ############### write code for 1 RNN layer (GRU) + 1 dense layer ###############

    return logits, placeholders


# train ops for training models
def backward_path(logits, labels):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='loss'
    )
    cost = tf.reduce_mean(cost, name='cost')
    train = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost, name="train")
    return cost, train


# performance measures for training/test set
def performance(logits, labels):
    prob = tf.nn.sigmoid(logits)
    prediction_binary = tf.cast(prob > .5, tf.float32)
    labels = tf.cast(labels, tf.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction_binary, labels), dtype=tf.float32), name="acc")
    return acc


# wrapper
def main():
    train_X, train_Y, test_X, test_Y = read_data()
    train_X_padded, train_X_lengths = preprocsesing(train_X)
    test_X_padded, test_X_lengths = preprocsesing(test_X)

    hyperparams = {
        'embedding_dims': 100,
        'max_seq_len': 100,
        'n_of_unique_words': np.max(train_X_padded)+1,
        'n_of_hidden_units': 50,
    }

    # build graph
    logits, placeholders = forward_path(train_X_padded, train_X_lengths, test_X_padded, test_X_lengths, hyperparams)
    input_sentence_idx, input_sentence_len, labels, batch_size_holder = placeholders
    cost, train = backward_path(logits, labels)
    accuracy = performance(logits, labels)

    # run models
    number_of_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # model training
        batch_size = 100
        for i in range(number_of_epochs):
            current_cost = 0
            current_accuracy = 0
            for j in range(int(train_X.shape[0]/batch_size)):
                batch_X = train_X_padded[j*batch_size : (j+1)*batch_size]
                batch_Y = train_Y[j*batch_size : (j+1)*batch_size]
                batch_len = train_X_lengths[j*batch_size : (j+1)*batch_size]

                c, _, a = sess.run(
                    [cost, train, accuracy],
                    feed_dict={
                        input_sentence_idx:batch_X,
                        input_sentence_len:batch_len,
                        labels:batch_Y,
                        batch_size_holder:batch_size
                        }
                    )

                current_cost += c / int(train_X.shape[0]/batch_size)
                current_accuracy += a / int(train_X.shape[0]/batch_size)

            print("epoch:", i+1, "cost:", current_cost, "training set accuracy:", current_accuracy)

        # model testing
        batch_size = test_X.shape[0]
        test_acc = sess.run(
            accuracy,
            feed_dict={
                input_sentence_idx:test_X_padded,
                input_sentence_len:test_X_lengths,
                labels:test_Y,
                batch_size_holder:batch_size
                }
            )
        print("Result: test set accuracy:", test_acc)

    return tf.trainable_variables(), test_acc

if __name__ == '__main__':
    main()
