from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import elice_utils
import csv

def getNumNoises(filename) :
    '''
    csv 형식의 파일 filename이 주어집니다.
    이 때, Grocery와 Milk만을 고려하였을 때의 노이즈 개수를 반환하세요.
    '''
    data = []
    csvreader = csv.reader(open(filename))
    for line in csvreader:
        data.append([line[4], line[3]])
    # reomove tag
    del data[0]

    db = DBSCAN(eps=2500, min_samples = 4).fit(data)

    return sum(db.labels_)*-1

print(getNumNoises("data.csv"))
