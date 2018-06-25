import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import elice_utils
from hierarchical_utils import agglomerative_cluster, distance
from matplotlib.animation import FuncAnimation

def main():
    data = np.array(read_data("data.txt"))
    '''
    distance는 distance() 함수를 argument로 보내는 것입니다. 다음 문제에서 agglomerative_cluster 함수를 직접 만들어 보겠습니다.
    '''
    cluster_histories = agglomerative_cluster(data, distance)

    '''
    여기서부터 애니메이션 그래프를 생성하는 코드입니다.
    '''
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    '''
    각 프레임별로 색깔을 어떻게 세팅할지 정의합니다.
    '''
    def get_colors(clusters):
        colors = [0] * len(data)
        for i in range(len(clusters)):
            for pi in clusters[i]:
                colors[pi] = i
        return colors

    '''
    첫 번째 프레임을 만듭니다.
    '''
    ax.scatter(data[:,0], data[:,1], c=get_colors(cluster_histories[0]))

    '''
    i번째 프레임을 어떻게 업데이트할지 정의합니다.
    '''
    def update(i):
        label = 'timestep {0}'.format(i)
        ax.scatter(data[:,0], data[:,1], c=get_colors(cluster_histories[i]))
        ax.set_title(label)
        return ax

    '''
    애니메이션을 만듭니다.
    '''
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(data)), interval=1000)
    anim.save('line.gif', dpi=80, writer='imagemagick')
    elice_utils.send_image('line.gif')

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
