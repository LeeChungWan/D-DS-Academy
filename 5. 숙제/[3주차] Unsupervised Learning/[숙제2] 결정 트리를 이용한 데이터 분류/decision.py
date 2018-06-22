from sklearn import tree
import csv

def main():
    data, target = read_data()
    clf = learn_decision_tree(data, target)
    data = input_data()
    print(classify(clf, data))

def read_data():
    fish_data = []
    fish_target = []

    '''
    1. fish.csv 파일을 읽어 무게와 비늘 밝기는 fish_data에, 연어인지 여부는 fish_target에 채워 넣습니다.
    '''
    with open("./fish.csv") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            attributes = line.split(",")
            fish_data.append([float(attributes[0]), float(attributes[1])])
            if attributes[2] == 'sea_bass':
                fish_target.append(0)
            else:
                fish_target.append(1)
    return (fish_data, fish_target)

def learn_decision_tree(data, target):
    '''
    2. sklearn.tree.DecisionTreeClassifier 기능을 사용해 데이터를 학습합니다.
    '''
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    return clf

def input_data():
    data = []

    '''
    input() 함수를 이용해 stdin 으로 입력을 받습니다.
    '''
    number_of_inputs = int(input())
    for i in range(number_of_inputs):
        line = input()
        data.append([float(line.split()[0]), float(line.split()[1])])
    print(data)
    return data

def classify(clf, data):
    predicted_classes = []
    classes = ['sea bass', 'salmon']

    '''
    3. 새로운 결제 건들을 입력받아 정상적인 데이터인지 여부를 판별합니다.
    판별 후, 데이터 포인트 하나 당 'sea bass' 혹은 'salmon' 문자열을 predicted_classes 리스트에 추가합니다.
    '''
    for i in range(len(data)):
        result = clf.predict(data[i])
        predicted_class = result[0]
        if predicted_class == 1:
            predicted_classes.append(classes[1])
        else:
            predicted_classes.append(classes[0])
    return predicted_classes

if __name__ == "__main__":
    main()
