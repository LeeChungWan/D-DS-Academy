import io
import numpy
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import elice_utils

special_chars_remover = re.compile("[^\w'|_]")
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

def main():
    training_sentences = read_data()
    testing_sentence = "어설픈 연기들로 몰입이 전혀 안되네요"
    prob_pair = naive_bayes(training_sentences, testing_sentence)

    plot_title = testing_sentence
    if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    visualize_boxplot(plot_title,
                  list(prob_pair),
                  ['Negative', 'Positive'])

def naive_bayes(training_sentences, testing_sentence):
    log_prob_negative = calculate_doc_prob(training_sentences[0], testing_sentence, 0.1) + math.log(0.5)
    log_prob_positive = calculate_doc_prob(training_sentences[1], testing_sentence, 0.1) + math.log(0.5)
    prob_pair = normalize_log_prob(log_prob_negative, log_prob_positive)

    return prob_pair

def read_data():
    training_sentences = [[], []]

    '''
    문제 1
    여기서 파일을 읽어 training_sentences에 저장합니다.
    '''
    tokens = []
    with open("./ratings.txt") as fp:
        tokens = fp.readlines()
    tokens = [x.strip() for x in tokens]

    for i in range(1, len(tokens)):
        token = tokens[i].split('\t')
        if token[2] is '0':
            training_sentences[0].append(token[1])
        else:
            training_sentences[1].append(token[1])
    return [' '.join(training_sentences[0]), ' '.join(training_sentences[1])]

def normalize_log_prob(prob1, prob2):

    '''
    문제 4
    로그로 된 확률값을 표준화합니다.
    이 부분은 이미 작성되어 있습니다.
    '''

    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)

def calculate_doc_prob(training_sentence, testing_sentence, alpha):
    logprob = 0

    training_model = create_BOW(training_sentence)
    testing_model = create_BOW(testing_sentence)
    '''
    문제 3
    training_sentence로 만들어진 모델이,
    testing_sentence를 만들어 낼 **로그 확률** 을 구합니다.
    일반 숫자에서 로그값을 만들기 위해서는 math.log() 를 사용합니다.

    일반 숫자에서의 곱셈이 로그에서는 덧셈, 나눗셈은 뺄셈이 된다는 점에 유의하세요.
    예) 3 * 5 = 15
        log(3) + log(5) = log(15)

        5 / 2 = 2.5
        log(5) - log(2) = log(2.5)
    '''
    # 각가의 token의 개수의 합계 구하기.
    total_size = 0
    for key in training_model:
        total_size += training_model[key]
    print(total_size)
    # P(testing | training) 구하기.
    for key in testing_model:
        if key in training_model:
            logprob += math.log(training_model[key] / total_size)
        else:
            logprob += math.log(alpha / total_size)
    return logprob

def create_BOW(sentence):
    bow = {}

    '''
    문제 2
    이전 실습과 동일하게 bag of words를 만듭니다.
    '''

    # 단어는 공백으로 나뉘어지며 모두 소문자로 치환되어야 합니다.
    sentence_lowered = sentence.lower()
    # 특수문자를 모두 제거합니다.
    sentence_without_special_characters = remove_special_characters(sentence_lowered)
    # 단어는 space를 기준으로 잘라내어 만듭니다.
    splitted_sentence = sentence_without_special_characters.split()
    # 단어는 한 글자 이상이어야 합니다.
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) >= 1
    ]

    # bow에 개수에 맞게 추가해줍니다.
    for token in splitted_sentence_filtered:
        bow.setdefault(token, 0)
        bow[token] += 1

        '''
        if token not in bow:
            bow[token] = 1
        else:
            bow[token] += 1
        '''
    return bow

'''
이 밑의 코드는 시각화를 위한 코드입니다.
궁금하다면 살펴보세요.
'''
def visualize_boxplot(title, values, labels):
    width = .35

    print(title)

    fig, ax = plt.subplots()
    ind = numpy.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

    autolabel(rects)

    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == "__main__":
    main()
