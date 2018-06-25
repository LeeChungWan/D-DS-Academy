from sklearn import tree
import pydotplus
import elice_utils

students_data = []
students_target = []

# 1

f = open('./students_train.csv', 'r')
for line in f:
    student = line.strip().split(',')
    students_data.append([
        float(student[0]),
        1 if student[2] == 'female' else 0,
        1 if student[3] == 'white' else 0,
        1 if student[3] == 'black' else 0,
        1 if student[3] == 'hispanic' else 0
    ])
    students_target.append(1 if float(student[1]) >= 185 else 0)
f.close()

# 2

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(students_data, students_target)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['weight', 'is_woman', 'is_white', 'is_black', 'is_hispanic'],
                                class_names=['shorter_than_185', 'taller_than_185']
                               )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("graph.png")
elice_utils.send_image("graph.png")
