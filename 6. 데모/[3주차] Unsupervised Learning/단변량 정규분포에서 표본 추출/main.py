import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils

def read_input():
	m, s = [int(x) for x in input().split()]
	return m, s

def generate_samples(m, s):
	return np.random.normal(m, s, 1000)

def plotting(samples, m, s):
	count, bins, ignored = plt.hist(samples, 30, normed=True)
	plt.plot(bins, norm.pdf(bins, m, s), linewidth=2, color='r')
	plt.xlabel("X", fontsize=20)
	plt.ylabel("P(X)", fontsize=20)
	plt.xlim(0, 100)
	plt.savefig("normal_samples.png")
	elice_utils.send_image("normal_samples.png")
	plt.close()

	return

def main():
	m, s = read_input()
	samples = generate_samples(m, s)
	plotting(samples, m, s)
	return

if __name__ == '__main__':
	main()
