import re

def main():
    sentence = input("입력 > ")
    BOW_dict, BOW = create_BOW(sentence)

    print(BOW_dict)
    print(BOW)

def create_BOW(sentence):
    bow_dict = {}
    bow = []
    # 단어를 소문자로 통일.
    sentence_lowered = sentence.lower()
    # 특수 문제 제거.
    sentence_without_special_characters = replace_non_alphabetic_chars_to_space(sentence_lowered)
    # space를 기준으로 나누어 준다.
    splitted_sentence = sentence_without_special_characters.split()
    # 한 글자 미만인 공백은 제거.
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) >= 1
    ]
    index = 0
    for token in splitted_sentence_filtered:
        if token not in bow_dict:
            bow_dict[token] = index
            index += 1

    sentence_match_dict = []
    for key in splitted_sentence_filtered:
        sentence_match_dict.append(bow_dict[key])
    for i in range(len(bow_dict)):
        bow.append(sentence_match_dict.count(i))
    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()
