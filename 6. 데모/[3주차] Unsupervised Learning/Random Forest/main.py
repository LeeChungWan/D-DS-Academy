import matplotlib
matplotlib.use('Agg')

import elice_utils
import matplotlib.pyplot as plt
import random
import numpy as np

# classifier를 위한 sklearn의 ensemble, tree를 불러옵니다.
from sklearn import ensemble, tree

fish_data = []
fish_target = []

# 1. 이전 문제에서 작성한 fish.csv 파일을 읽는 부분을 그대로 사용하실 수 있습니다.
f = open('./fish.csv', 'r')
for line in f:
    # 여기에 코드를 채워넣어 주세요. (이전 문제에서 작성한 코드를 그대로 사용할 수 있습니다.)
    split_line = line.split(',')
    weight = float(split_line[0])
    lightness = float(split_line[1])
    if split_line[2].strip() == 'salmon':
        is_salmon = 1
    else:
        is_salmon = 0

    fish_data.append([weight, lightness])
    fish_target.append(is_salmon)

fish_data = fish_data[:40]
fish_target = fish_target[:40]
f.close()

# 2. 이전 문제에서 사용한 sklearn.tree.DecisionTreeClassifier 기능을 사용해 데이터를 학습합니다.

# 여기에 코드를 채워넣어 주세요. (이전 문제에서 작성한 코드를 그대로 사용할 수 있습니다.)
clf_decision_tree = tree.DecisionTreeClassifier()
clf_decision_tree = clf_decision_tree.fit(fish_data, fish_target)

# 3. sklearn.ensemble.RandomForestClassifier 기능을 사용해 데이터를 학습합니다.

# 여기에 코드를 채워넣어 주세요.
clf_random_forest = ensemble.RandomForestClassifier()
clf_random_forest = clf_random_forest.fit(fish_data, fish_target)


# 4. 아래의 코드는 각 classifier의 decision boundary를 그려주는 코드입니다.
#    해당 코드는 이미 완성되어 있으므로 수정하실 필요가 없습니다.
plot_step = 0.02
plot_colors = "rb"
cmap = plt.cm.RdYlBu

# weight를 그래프의 x축으로 사용합니다.
x_min = min(fish_data, key=lambda weight_lightness: weight_lightness[0])[0]
x_max = max(fish_data, key=lambda weight_lightness: weight_lightness[0])[0]

# lightness를 그래프의 y축으로 사용합니다.
y_min = min(fish_data, key=lambda weight_lightness: weight_lightness[1])[1]
y_max = max(fish_data, key=lambda weight_lightness: weight_lightness[1])[1]

# (x_min, y_min) ~ (x_max, y_max)의 사각형 공간에 plot_step 간격으로 점을 찍은 후,
# 각 점의 classify 결과를 바탕으로 그래프에 색을 지정합니다.
# xx, yy는 각각 해당 사각형 공간에 plot_step 간격으로 찍힌 점들의 위치를 나타냅니다.
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

# 세 종류의 그래프를 그립니다.
for plot_idx, (model_name, model) in enumerate([('Decision Tree', clf_decision_tree),
                                                ('Random Forest', clf_random_forest),
                                                ('Random Forest (Voting)', clf_random_forest)],
                                               start=1):
    # 그래프의 위치와 제목을 지정합니다.
    plt.subplot(1, 3, plot_idx)
    plt.title(model_name)

    # Decision boundary를 표시합니다.
    if plot_idx == 1 or plot_idx == 3:
        # 그래프의 각 위치에 대해 binary classify 한 결과를 구합니다.
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # classify 결과에 따라 색을 지정합니다.
        cs = plt.contourf(xx, yy, Z, cmap=cmap)
    else:
        # 각 tree에서 구한 binary classify 결과를 1 / (트리의 갯수) 의 가중치로 합하여 사용합니다.
        estimator_alpha = 1.0 / len(model.estimators_)

        # Ensemble에 사용된 여러개의 tree에 대해, 각각의 결과를 aggregate한 결과를 사용합니다.
        for tree in model.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

    # 학습에 사용한 점들을 표시합니다.
    for i in range(len(fish_data)):
        x, y = fish_data[i]
        plt.scatter(x, y,
                    c=plot_colors[fish_target[i]],
                    cmap=cmap)

plt.savefig("image.png")
elice_utils.send_image("image.png")
