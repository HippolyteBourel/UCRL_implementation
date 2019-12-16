from learners.UCRL2_local2 import *
from learners.UCRL2_local import *
import scipy as sp
import numpy as np
import math


class UCRL2_Bernstein(UCRL2_local2):
	def name(self):
		return "UCRL2_Bernstein"
	
	def beta(self, n, delta):
		temp = 2 * np.log(np.log(max((np.exp(1), n)))) + np.log(3 / delta)
		return temp

	# A difference with "real" bernstein bounds: here we fix r_max to 1 (it is the case in our test base) update 19/09/19: from ALT
	def distances(self):
		delta = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(self.r_estimate[s, a] * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
				for next_s in range(self.nS):
					p = self.p_estimate[s, a, next_s]
					temp = np.sqrt(p * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
					self.p_distances[s, a, next_s] = temp

class UCRL2_Bernstein_old(UCRL2_local2):
	def name(self):
		return "UCRL2_Bernstein(old)"

	# A difference with "real" bernstein bounds: here we fix r_max to 1 (it is the case in our test base)
	def distances(self):
		c = 1.1
		delta = self.delta
		bk = np.log(2 * self.nS * self.nA * self.t / self.delta)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				tau = np.log((1 / delta) * math.ceil((np.log(n * np.log(c * n)) / np.log(c))))
				b2 = 8 * tau / 3
				r = self.r_estimate[s, a]
				temp = r * (1 - r) + 2 * np.sqrt((b2 * r * (1 - r)) / n) + (7 * b2) / n
				self.r_distances[s, a] = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n
				for next_s in range(self.nS):
					p = self.p_estimate[s, a, next_s]
					temp = p * (1 - p) + 2 * np.sqrt((b2 * p * (1 - p)) / n) + (7 * b2) / n
					self.p_distances[s, a, next_s] = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n


class UCRL2_Bernstein2(UCRL2_local2):
	def name(self):
		return "UCRL2_Bernstein"

	# A difference with "real" bernstein bounds: here we fix r_max to 1 (it is the case in our test base)
	def distances(self):
		c = 1.1
		delta = self.delta
		bk = np.log(2 * self.nS * self.nA * self.t / self.delta)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				tau = np.log((1 / delta) * math.ceil((np.log(n * np.log(c * n)) / np.log(c))))
				b2 = 8 * tau / 3
				r = self.r_estimate[s, a]
				temp = p * (1 - p) + 2 * np.sqrt((b2 * r * (1 - r)) / n) + (7 * b2) / n
				self.r_distances[s, a] = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n
				for next_s in range(self.nS):
					p = self.p_estimate[s, a, next_s]
					temp = p * (1 - p) + 2 * np.sqrt((b2 * p * (1 - p)) / n) + (7 * b2) / n
					self.p_distances[s, a, next_s] = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n

# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
class UCRL2_Bernstein_old(UCRL2_local):
	def name(self):
		return "UCRL2_Bernstein_old"

	# Auxiliary function to compute the empirical variance for r(s, a) and p(s, a, s').
	def variances(self):
		T = len(self.observations[1])
		if T == 0: # To initialize these values and later just update them instead of computing from scratch at each episode
			self.tk = 0
			self.r_variance = np.zeros((self.nS, self.nA))
			self.p_variance = np.zeros((self.nS, self.nA, self.nS))
		for t in range(self.tk, T):
			s = self.observations[0][t]
			a = self.observations[1][t]
			next_s = self.observations[0][t + 1]
			self.r_variance[s, a] += (1 / max((1, self.Nk[s, a]))) * (self.observations[2][t] - self.r_estimate[s, a])**2
			for ss in range(self.nS):
				obs = 1 if ss == next_s else 0
				self.p_variance[s, a, next_s] += (1 / max((1, self.Nk[s, a]))) * (obs - self.p_estimate[s, a, ss])**2# To complete
		self.tk = T
		return self.r_variance, self.p_variance

	# A difference with "real" bernstein bounds: here we fix r_max to 1 (it is the case in our test base)
	def distances(self):
		r_variance, p_variance = self.variances()
		bk = np.log(2 * self.nS * self.nA * self.t / self.delta)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				nm1 = max(1, self.Nk[s, a] - 1)
				self.r_distances[s, a] = np.sqrt((14 * r_variance[s, a] * bk) / n) + (49 / 3 * bk) / nm1
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt((14 * p_variance[s, a, next_s] * bk) / n) + (49 / 3 * bk) / nm1
					


class UCRL2_Bernstein2_old(UCRL2_local2):
	def name(self):
		return "UCRL2_Bernstein2_old"

	# Auxiliary function to compute the empirical variance for r(s, a) and p(s, a, s').
	def variances(self):
		T = len(self.observations[1])
		if T == 0: # To initialize these values and later just update them instead of computing from scratch at each episode
			self.tk = 0
			self.r_variance = np.zeros((self.nS, self.nA))
			self.p_variance = np.zeros((self.nS, self.nA, self.nS))
		for t in range(self.tk, T):
			s = self.observations[0][t]
			a = self.observations[1][t]
			next_s = self.observations[0][t + 1]
			self.r_variance[s, a] += (1 / max((1, self.Nk[s, a]))) * (self.observations[2][t] - self.r_estimate[s, a])**2
			for ss in range(self.nS):
				obs = 1 if ss == next_s else 0
				self.p_variance[s, a, next_s] += (1 / max((1, self.Nk[s, a]))) * (obs - self.p_estimate[s, a, ss])**2# To complete
		self.tk = T
		return self.r_variance, self.p_variance

	# A difference with "real" bernstein bounds: here we fix r_max to 1 (it is the case in our test base)
	def distances(self):
		r_variance, p_variance = self.variances()
		bk = np.log(2 * self.nS * self.nA * self.t / self.delta)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				nm1 = max(1, self.Nk[s, a] - 1)
				self.r_distances[s, a] = np.sqrt((14 * r_variance[s, a] * bk) / n) + (49 / 3 * bk) / nm1
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt((14 * p_variance[s, a, next_s] * bk) / n) + (49 / 3 * bk) / nm1