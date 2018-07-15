import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

PATH = "./data/"

def load_data(n_of_training_ex, n_of_testing_ex):
    X_train = np.load(PATH + "X_train.npy")[:n_of_training_ex]
    y_train = np.load(PATH + "y_train.npy")[:n_of_training_ex]
    X_test = np.load(PATH + "X_test.npy")[:n_of_testing_ex]
    y_test = np.load(PATH + "y_test.npy")[:n_of_testing_ex]
    return X_train, y_train, X_test, y_test

def train(X_train, y_train):
    # truncate and pad input sequences
    max_review_length = 300
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

    # create the model
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(1000, embedding_vector_length, input_length=max_review_length))

    model.add(SimpleRNN(50))
    #model.add(LSTM(50))
    #model.add(GRU(50))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)
    return model

def test(model, X_test, y_test):
    # Final evaluation of the model
    max_review_length = 300
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    scores = model.evaluate(X_test, y_test, verbose=0)
    return scores

def main():
    n_of_training_ex = 2000
    n_of_testing_ex = 300

    X_train, y_train, X_test, y_test = load_data(n_of_training_ex, n_of_testing_ex)

    model = train(X_train, y_train)

    scores = test(model, X_test, y_test)
    print("Test Set Accuracy: %.2f%%" % (scores[1]*100))

    return

if __name__ == '__main__':
    main()
