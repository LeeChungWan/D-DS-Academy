import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
import re
from scipy.spatial.distance import cosine
from sklearn.decomposition import randomized_svd

def main():
    data = read_data("data/data.txt")
    stopwords = read_data("data/stopwords.txt")
    bow, vocab = construct_bag_of_words(data,stopwords)
    Documents,Words = LSA(bow)
    plot(Documents, Words, vocab)
    answer = query(Documents,Words)
    print("""Closest documents to the query "dagger die" are: """)
    for x in answer:
        print(x[0]," : ",x[1])
    return answer

def construct_bag_of_words(data,stopwords):
    # Write here

    vocab = []
    for i in range(len(data)):
        # 단어는 공백으로 나뉘어지며 모두 소문자로 치환되어야 합니다.
        sentence_lowered = data[i].lower()
        # 특수문자를 모두 제거합니다.
        sentence_without_special_characters = replace_non_alphabetic_chars_to_space(sentence_lowered)
        # 단어는 space를 기준으로 잘라내어 만듭니다.
        splitted_sentence = sentence_without_special_characters.split(' ')
        # 단어는 한 글자 이상이어야 합니다.
        splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) > 1 and token not in stopwords
        ]
        for token in splitted_sentence_filtered:
            if token not in vocab:
                vocab.append(token)
        data[i] = splitted_sentence_filtered

    bow = [[0 for j in range(len(vocab))] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(vocab)):
            for k in range(len(data[i])):
                if data[i][k] == vocab[j]:
                    bow[i][j] += 1
    return bow,vocab

def LSA(bow):
    # Write here
    X = np.asarray(bow)
    #Computing SVD
    U, Sigma, VT = randomized_svd(X, n_components=2, n_iter=1, random_state=None)
    Sigma=np.diag(Sigma)
    W = np.dot(Sigma,VT)
    D = np.dot(U,Sigma)
    return D, W

def query(D, W):
    query=(W.T[3]+W.T[5])/2
    n2=["D1", "D2", "D3", "D4", "D5"]
    results=[]
    for i in range(len(D)):
        results.append((n2[i],cosine(D[i],query)))
    return sorted(results, key=lambda tup: tup[1])

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub("[^-a-zA-Z]+", " ", sentence)

def plot(D, W, n1):
    n2=["D1", "D2", "D3", "D4", "D5"]
    fig, ax = plt.subplots()
    ax.scatter(W[0], W[1])
    ax.scatter(D.T[0], D.T[1])
    for i, txt in enumerate(n1):
        if(txt=="free"):
            ax.annotate(txt, (W[0][i]-0.13,W[1][i]-0.02))
        elif txt=="motto":
            ax.annotate(txt, (W[0][i]-0.06,W[1][i]+0.07))
        elif txt=="know":
            ax.annotate(txt, (W[0][i]-0.14,W[1][i]+0.03))
        elif txt=="live":
            ax.annotate(txt, (W[0][i]+0.03,W[1][i]-0.03))
        else:
            ax.annotate(txt, (W[0][i]+0.02,W[1][i]+0.02))

    for i, txt in enumerate(n2):
        ax.annotate(txt, (D.T[0][i]+0.02,D.T[1][i]+0.03))

    plt.savefig('demo.png')
    elice_utils.send_image('demo.png')

def read_data(filename):
    data = []
    with open(filename) as fp:
        for line in fp:
            data.append(line.strip("\n"))

    return data

if __name__ == '__main__':
    main()
