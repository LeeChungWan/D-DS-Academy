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
    variance = np.var(data)
    return (average, variance)

def doSampleAndMultiplication(mu1, sigma1, mu2, sigma2, n) :
    '''
    평균이 mu1, 분산이 sigma1^2인 가우시안 분포 f와
    평균이 mu2, 분산이 sigma2^2인 가우시안 분포 g로부터 각각 표본을 추출하여
    그들의 곱을 리스트로 반환하는 함수를 작성하세요.
    '''

    result = []
    f = np.random.normal(mu1, sigma1, n)
    g = np.random.normal(mu2, sigma2, n)
    result = f * g
    return result

def plotResult(data) :
    '''
    숫자들로 이루어진 리스트 data가 주어질 때, 이 data의 분포를 그래프로 나타냅니다.

    이 부분은 수정하지 않으셔도 됩니다.
    '''

    filename = "uniform.png"

    frequency = [ 0 for i in range(int(max(data))+1) ]

    for element in data :
        frequency[int(element)] += 1

    n = len(frequency)

    myRange = range(0, len(frequency))
    width = 1

    plt.bar(myRange, frequency, width, color="blue")

    plt.xlabel("Sample", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()

def main():
    '''
    이 부분은 수정하지 않으셔도 됩니다.
    '''

    line1 = [int(x) for x in input("입력 >").split()]
    line2 = [int(x) for x in input("입력 >").split()]
    line3 = [int(x) for x in input("입력 >").split()]

    mu1 = line1[0]
    sigma1 = line1[1]

    mu2 = line2[0]
    sigma2 = line2[1]

    n = line3[0]

    result = doSampleAndMultiplication(mu1, sigma1, mu2, sigma2, n)

    plotResult(result)

    stat = getStatistics(result)

    print("average : %.2lf" % stat[0])
    print("variance : %.2lf" % stat[1])

if __name__ == "__main__":
    main()
