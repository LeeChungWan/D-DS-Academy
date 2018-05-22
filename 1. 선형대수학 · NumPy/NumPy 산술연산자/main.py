import numpy as np

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = np.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # 아래 코드를 작성하세요.
    # 1. A 원소의 합이 1이 되로록 표준화를 적용하고 결과값을 A에 저장.
    A = A / A.sum()
    # 2. matrix_tutorial() 함수가 A의 분산 값을 리턴하도록 변경.
    return A.var()

if __name__ == "__main__":
    main()
