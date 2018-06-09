from scipy.stats import linregress
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
import math
import operator

def main():
    words = read_data()
    words = sorted(words, key = lambda words:words[1], reverse = True)

    X = list(range(1, len(words)+1))
    Y = [x[1] for x in words]

    X, Y = np.array(X), np.array(Y)
    X, Y = np.log(X), np.log(Y)
    slope, intercept = do_linear_regression(X, Y)
    draw_chart(X, Y, slope, intercept)
    return slope, intercept

def read_data():
    words = []
    with open("words.txt") as f:
        lines = f.readlines()
        for line in lines:
            word = line.split(',')[0]
            frequency = line.split(',')[1]
            words.append([word, int(frequency)])
    return words

def draw_chart(X, Y, slope, intercept):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X, Y)

    # 차트의 X, Y축 범위와 그래프를 설정합니다.
    min_X = min(X)
    max_X = max(X)
    min_Y = min_X * slope + intercept
    max_Y = max_X * slope + intercept
    plt.plot([min_X, max_X], [min_Y, max_Y],
             color='red',
             linestyle='--',
             linewidth=3.0)

    # 기울과와 절편을 이용해 그래프를 차트에 입력합니다.
    ax.text(min_X, min_Y + 0.1, r'$y = %.2lfx + %.2lf$' % (slope, intercept), fontsize=15)

    plt.savefig('chart.png')
    elice_utils.send_image('chart.png')

def do_linear_regression(X, Y):
    slope, intercept, r_value, p_value, std_err = linregress(X,Y)
    return slope, intercept

if __name__ == "__main__":
    main()
