import operator
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils


def main():
    words = read_data()
    # lambda를 사용하면 복잡한 형태의 list에서 정렬 기준을 정할 수 있다.
    words = sorted(words,key = lambda words: words[1], reverse = True) # words.txt 단어를 빈도수 순으로 정렬합니다.

    # 정수로 표현된 단어를 X축 리스트에, 각 단어의 빈도수를 Y축 리스트에 저장합니다.
    X = list(range(1, len(words)+1))
    Y = [x[1] for x in words]

    # X, Y 리스트를 array로 변환한 후 각 원소 값에 log()를 적용합니다.
    X, Y = np.array(X), np.array(Y)
    X, Y = np.log(X), np.log(Y)
    # 기울기와 절편을 구한 후 그래프와 차트를 출력합니다.
    slope, intercept = do_linear_regression(X, Y)
    draw_chart(X, Y, slope, intercept)

    return slope, intercept


# read_data() - words.txt에 저장된 단어와 해당 단어의 빈도수를 리스트형으로 변환합니다.
def read_data():
    # words.txt 에서 단어들를 읽어,
    # [[단어1, 빈도수], [단어2, 빈도수] ... ]형으로 변환해 리턴합니다.
    words = []
    f = open("./words.txt")
    while True:
        line = f.readline()
        if not line: break
        word = line.split(',')[0]
        frequency = line.split(',')[1]
        words.append([word, float(frequency)])

    return words


# do_linear_regression() - 임포트한 sklearn 패키지의 함수를 이용해 그래프의 기울기와 절편을 구합니다.
def do_linear_regression(X, Y):
    # do_linear_regression() 함수를 작성하세요.
    X = X.reshape(-1, 1)    # reshape 필수!
    lrmodel = LinearRegression()
    lrmodel.fit(X, Y)
    slope = lrmodel.coef_[0]
    intercept = lrmodel.intercept_
    print(slope)
    print(intercept)
    return (slope, intercept)


# draw_chart() - matplotlib을 이용해 차트를 설정합니다.
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

if __name__ == "__main__":
    main()
