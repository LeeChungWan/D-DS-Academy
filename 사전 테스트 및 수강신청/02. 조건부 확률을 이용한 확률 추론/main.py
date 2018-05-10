from elice_utils import EliceUtils

elice_utils = EliceUtils()

def calculate_P_A(prob_single_event, prob_table):
    P_A = 0
    P_B = prob_single_event[0]
    P_E = prob_single_event[1]
    P_G = prob_single_event[2]
    for row in prob_table:
        B = row[0]
        E = row[1]
        G = row[2]
        T = row[3]
        num = 1
        if B == 1:
            num *= P_B
        else:
            num *= 1-P_B
        if E == 1:
            num *= P_E
        else:
            num *= 1-P_E
        if G == 1:
            num *= P_G
        else:
            num *= 1-P_G
        num *= T
        P_A += num

    # 여기에 정답을 구현해 주세요.
        
    return round(P_A,4)

# 아래 부분은 함수의 호출과 출력을 위한 부분입니다. 수정하지 마세요

def read_inputs():
    prob_single_event = [float(i) for i in input().split()]
    prob_table = []
    for i in range(2 ** len(prob_single_event)):
        prob_table.append(input().split())
    prob_table = [list(map(float, each)) for each in prob_table]
    return prob_single_event, prob_table


def main():
    prob_single_event, prob_table = read_inputs()
    ans = calculate_P_A(prob_single_event, prob_table)
    print(ans)


if __name__ == '__main__':
    main()

