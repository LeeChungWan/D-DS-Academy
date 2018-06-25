import numpy as np
import copy

import elice_utils

def main():
    data = np.array(read_data("data.txt"))
    cluster_history = agglomerative_cluster(data, distance)

def agglomerative_cluster(data, distance_fn):
    cluster_history = []
    clusters = []

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

def distance(data, c1, c2):
    '''
    주어진 data의 c1번째 클러스터와 c2번째 클러스터의
    single-linkage clustering 위한 거리를 계산해서 리턴합니다.
    즉 c1에 있는 모든 데이터 포인트와 c2에 있는 모든 데이터 포인트의
    조합 중 가장 거리가 가까운 페어를 찾아 그 거리를 리턴합니다.
    거리는 유클리디언 거리를 사용합니다.
    '''
    min_distance = 100000
    for i in range(len(c1)):
        for j in range(len(c2)):
            dist = np.sqrt(np.sum((data[c1[i]] - data[c2[j]]) ** 2))
            min_distance = min(min_distance, dist)
    return min_distance


    #return np.sqrt(np.sum((x1 - x2) ** 2))

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
