import numpy as np
import copy

import elice_utils
from hierarchical_utils import distance

def main():
    data = np.array(read_data("data.txt"))
    cluster_history = agglomerative_cluster(data, distance)

def agglomerative_cluster(data, distance_fn):
    cluster_history = []
    clusters = []

    '''
    여기에 agglomerative clustering 방법을 구현합니다.
    cluster_history 변수에 매 step마다 cluster들을 저장합니다.
    주의할 점은 cluster_history.append(copy.deepcopy(clusters)) 와 같이
    deep copy를 해야 한다는 것입니다.

    distance_fn의 사용법은 다음과 같습니다.
    dist = distance_fn(data, clusters[i1], clusters[i2])
    data는 agglomerative_cluster에 주어진 data를 그대로 사용하면 되고,
    clusters[i1], clusters[i2] 는 각각 데이터 포인트의 인덱스로 이루어진 array입니다.
    '''
    for i in range(len(data)):
        clusters.append([i])
    cluster_history.append(copy.deepcopy(clusters))
    while len(clusters) is not 1:
        min_dist_index_1 = 0
        min_dist_index_2 = 0
        compare_dist = 1000000
        for i in range(len(clusters)-1):
            for j in range(i+1, len(clusters)):
                dist = distance_fn(data, clusters[i], clusters[j])
                if dist < compare_dist:
                    compare_dist = dist
                    min_dist_index_1 = i
                    min_dist_index_2 = j
        clusters[min_dist_index_1] = clusters[min_dist_index_1] + clusters[min_dist_index_2]
        del clusters[min_dist_index_2]
        cluster_history.append(copy.deepcopy(clusters))
    return cluster_history

def read_data(filename):
    '''
    read_data 함수를 통해 filename 에 들어 있는 데이터들을 읽습니다.
    '''
    data = []

    with open(filename) as fp:
        for line in fp:
            splitted = line.split(',')
            x = float(splitted[0].strip())
            y = float(splitted[1].strip())

            data.append((x, y))

    return data

if __name__ == "__main__":
    main()
