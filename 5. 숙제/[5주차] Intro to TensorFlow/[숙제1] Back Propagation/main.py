import numpy as np

def sigmoid(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    # derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))

def getParameters(X, y) :
    '''
    X, y 를 가장 잘 설명하는 parameter (w1, w2, w3)를 반환하는 함수를 작성하세요. 여기서 X는 (x1, x2, x3) 의 list이며, y 는 0 혹은 1로 이루어진 list입니다. 예를 들어, X, y는 다음의 값을 가질 수 있습니다.

    X = [(1, 0, 0), (1, 0, 1), (0, 0, 1)]
    y = [0, 1, 1]
    '''
    learning_rate = 0.5
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape((len(y),1))
    w = np.array([1, 1, 1], dtype=np.float32).reshape((3,1))
    for step in range(5001):
        l1 = np.matmul(X, w)
        hypothesis = sigmoid(l1)
        diff = hypothesis - y

        # Back prob
        d_l1 = diff * sigmoid_prime(l1)
        d_w1 = np.matmul(X.T, d_l1)
        w -= learning_rate * d_w1
        cost = 0.5 * ((y - hypothesis)**2)
        if step % 100 == 0:
            print(w)

    result = []
    for value in w:
        if value > 5:
            result.append(1)
        else:
            result.append(0)
    return tuple(result)

def main():
    '''
    이 코드는 수정하지 마세요.
    '''

    X = [(1, 0, 0), (1, 0, 1), (0, 0, 1)]
    y = [0, 1, 1]


    # 아래의 예제 또한 테스트 해보세요.
    #X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    #y = [0, 0, 1, 1, 1, 1, 1, 1]

    # 아래의 예제를 perceptron이 100% training할 수 있는지도 확인해봅니다.
    #X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    #y = [0, 0, 0, 1, 0, 1, 1, 1]


    print(getParameters(X, y))

if __name__ == "__main__":
    main()
