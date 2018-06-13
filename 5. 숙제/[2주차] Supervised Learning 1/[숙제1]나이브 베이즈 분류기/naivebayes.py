import re
import math
import naivebayes_utils

special_chars_remover = re.compile("[^\w'|_]")
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

def main():
    training1_sentence = input()
    training2_sentence = input()
    testing_sentence = input()

    alpha = float(input())
    prob1 = float(input())
    prob2 = float(input())

    print(naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2))

def naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2):
    # Implement Naive Bayes Algorithm here...
    # Return normalized log probability
    # of p(training1_sentence|testing_sentence) and p(training2_sentence|testing_sentence)

    bow_train1 = create_BOW(training1_sentence)
    bow_train2 = create_BOW(training2_sentence)
    bow_test = create_BOW(testing_sentence)

    classify1 = math.log(prob1)
    classify2 = math.log(prob2)

    total_size_train1 = sum(bow_train1.values())
    total_size_train2 = sum(bow_train2.values())
    for key in bow_test.keys():
        for i in range(0, bow_test[key]):
            if key in bow_train1:
                classify1 += math.log(bow_train1[key] + alpha)
            else:
                classify1 += math.log(alpha)
            if key in bow_train2:
                classify2 += math.log(bow_train2[key] + alpha)
            else:
                classify2 += math.log(alpha)
            classify1 -= math.log(total_size_train1 + (len(bow_train1) * alpha))
            classify2 -= math.log(total_size_train2 + (len(bow_train2) * alpha))
    return normalize_log_prob(classify1, classify2)

def normalize_log_prob(prob1, prob2):
    return naivebayes_utils.normalize_log_prob(prob1, prob2)

def log_likelihood(training_model, testing_model, alpha):
    return naivebayes_utils.calculate_doc_prob(training_model, testing_model, alpha)

def create_BOW(sentence):
    # bag-of-words
    # 문장에 들어있는 각 단어를 key로, 해당 단어가 문장에 나타나는 빈도수를 value로 하는 dictionary를 반환합니다.
    # 예: {'elice':3, 'data':1, ...}
    # 단어는 공백으로 나뉘어지며 모두 소문자로 치환되어야 한다.
    bow = {}
    sentence_lowered = sentence.lower()
    # 특수문자를 모두 제거합니다.
    sentence_without_special_characters = remove_special_characters(sentence_lowered)
    # 단어는 한 글자 이상이어야 합니다.
    # 단어는 space를 기준으로 잘라내어 만듭니다.
    splitted_sentence = sentence_without_special_characters.split()
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) >= 1
    ]

    for token in splitted_sentence_filtered:
        bow.setdefault(token, 0)
        bow[token] += 1
    print(bow)
    return bow

if __name__ == "__main__":
    main()
