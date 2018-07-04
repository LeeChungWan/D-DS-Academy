def activation(weight, x):
    return sum(a*b for a,b in zip(weight, x))

def output(weight, x):
    a = activation(weight, x)
    b = 1
    if a + b > 0:
        return 1
    return 0

def main():
    weight = [1, 0.5, -0.7, 0.1]
    x = [0, 1, 1, 0]

    if output(weight, x) == 1:
        print("My neuron fired a signal")
    else:
        print("My neuron did not fire a signal")

if __name__ == "__main__":
    main()
