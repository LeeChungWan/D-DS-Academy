import matplotlib
matplotlib.use('Agg')

import elice_utils
import matplotlib.pyplot as plt
import random
import numpy as np

def getStatistics(data) :
    '''
    추출 결과가 리스트로 주어질 때, 추출된 결과의 평균과 분산을 반환하는 프로그램을 작성하세요.
    '''

    average = sum(data) / len(data)
    variance = np.var(np.array(data))
    return (average, variance)

def doSample(n, m) :
    '''
    1 ~ n 까지의 수를 균일 분포를 따라 m회 추출한 결과를 리스트로 반환하는 함수를 작성합니다.
    예를 들어, n = 10, m = 3 일 경우에는 1 ~ 10 까지의 수를 균일 분포에 따라 3회 추출하므로
    가능한 결과로 [1, 8, 5] 가 있을 수 있습니다. 물론, 추출을 할 때마다 그 결과가 달라질 수 있습니다.
    '''

    result = []
    for i in range(m):
        data = random.randrange(1,n)
        result.append(data)
    return result

def plotResult(data) :
    frequency = [ 0 for i in range(max(data)+1) ]

    for element in data :
        frequency[element] += 1

    n = len(frequency)

    myRange = range(1, n)
    width = 1

    plt.bar(myRange, frequency[1:])

    plt.xlabel("Sample", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    filename = "chart.svg"
    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()

def main():
    '''
    이 부분은 수정하지 않으셔도 됩니다.
    '''

    line = [int(x) for x in input("입력 > ").split()]

    n = line[0]
    m = line[1]

    result = doSample(n, m)

    plotResult(result)

    stat = getStatistics(result)

    print(str(stat[0]) + " " + str(stat[1]))

if __name__ == "__main__":
    main()
