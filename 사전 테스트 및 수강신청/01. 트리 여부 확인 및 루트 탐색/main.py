from elice_utils import EliceUtils

elice_utils = EliceUtils()

def is_tree(tree_candidate):
    check_tree = []
    #여기에 함수를 구현해 주세요
    if len(tree_candidate) == 0:
        check_tree.append([0, True])
    else:
        for line in tree_candidate:
            result = []
            if len(line) == 0:
                result = [0, True]
            # 입력이 홀수개
            elif len(line) % 2 == 1:
                result = [0, False]
            else:
                # 노드의 개수
                nodeList = list(set(line))
                # 간선의 개수는 Node - 1
                if len(line)/2 != len(nodeList) - 1:
                    result = [0, False]
                else:
                    checkDuplication = []
                    index = 0
                    for node in line:
                        if index % 2 == 1:
                            if node not in checkDuplication:
                                checkDuplication.append(node)
                                nodeList.remove(node)
                            else:
                                result = [0, False]
                        index += 1
                    if len(nodeList) == 1:
                        result = [nodeList[0], True]
                    else:
                        result = [0, False]
            check_tree.append(result)

    return check_tree


# 아래 부분은 함수의 호출과 출력을 위한 부분입니다. 수정하지 마세요

def read_inputs():
    tree_candidate = []
    while True:
        a = input()
        if a == '-1':
            break
        else:
            candidate = list(int(x) for x in a.split())
            tree_candidate.append(candidate)
    return tree_candidate


def main():
    tree_candidate = read_inputs()
    ans = is_tree(tree_candidate)
    print(ans)


if __name__ == "__main__":
    main()