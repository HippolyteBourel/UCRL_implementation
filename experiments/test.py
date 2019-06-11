import numpy as np
import math
import pylab as pl


T = 1000000
nS = 25
delta = 0.05
for t in range(1, T):
	temp = nS * (1 + math.ceil(np.log(2*T / t) / np.log(4/3))) * np.exp(- t)
	if temp <= delta:
		tau = t
		break
#print(tau)
#l = (4 / 3) * tau
p_estimate = 0

def beta_minus(p, n):
	d = 0.05 / (2 * 25 * 2)
	g = (1 / 2 - p) / np.log(1 / p - 1)
	return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

def beta_plus(p, n):
	d = 0.05 / (2 * 25 * 2)
	if p >= 0.5:
		g = p * (1 - p)
	else:
		g = (1 / 2 - p) / np.log(1 / p - 1)
	return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		

def bound_plus(p_est, n):
	p_tilde = p_est + np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
	up = p_tilde
	down = p_est
	for i in range(16):
		temp = (up + down) / 2
		if (temp - beta_minus(temp, n)) <= p_est:
			down = temp
		else:
			up = temp
	return (up + down) / 2 - p_est
		
def bound_minus(p_est, n):
	p_tilde = p_est - np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
	down = p_tilde
	up = p_est
	for i in range(16):
		temp = (up + down) / 2
		if (temp + beta_plus(temp, n)) >= p_est:
			up = temp
		else:
			down = temp
	return p_est - (up + down) / 2
	

"""c = 1.1
d = 0.05 / (2 * 25 * 2)
#li = [0.05, 0.1, 0.2, 1/2, 3/4, 5/6, 9/10, 0.95]
li = [0.5]
legend = []
for i in range(len(li)):
	X = []
	p = li[i]
	legend.append(str(p))
	if p >= 0.5:
		g = p * (1 - p)
	else:
		g = (1 / 2 - p) / np.log(1 / p - 1)
	for n in range(2, 1000):
		tau = np.log((1 / delta) * math.ceil((np.log(n * np.log(c * n)) / np.log(c))))
		b2 = 8 * tau / 3
		temp = p * (1 - p) + 2 * np.sqrt((b2 * p * (1 - p)) / n) + (7 * b2) / n
		v1 = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n
		#v1 = 100
		v2 = np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		if v2 < 0.1:
			print("Separation for n = ", n)
		X.append(min(v1, v2))
	pl.plot(X)"""
	

X = []
Z = []
Y = []
d = 0.05 / (2 * 25 * 2)
for n in range(1, 100000):
	X.append(bound_minus(1, n))
	Z.append(bound_plus(0, n))
	Y.append(np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n))

pl.plot(X)
pl.plot(Y)
pl.plot(Z)

pl.yscale('log')
#pl.legend(legend)
pl.xscale('log')
pl.show()

X = []
N = 1000000
for i in range(1, N - 1):
	p = i / N
	if p > 0.5:
		g = p * (1 - p)
	else:
		g = (1 / 2 - p) / np.log(1 / p - 1)
	X.append(g)
pl.figure()
pl.plot(X)
#pl.show()