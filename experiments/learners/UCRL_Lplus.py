from learners.UCRL2_local import *
import scipy as sp
import numpy as np

# Component-wise UCRL with tigth Laplace-improved confidence bounds
class UCRL2_Lplus(UCRL2_local):
	def __init__(self, nS, nA, delta):
		super().__init__(nS, nA, delta)
		self.p_distances = np.zeros((nS, nA, nS, 2))
	
	def name(self):
		return "UCRL2_Lplus"
	
	def beta_minus(self, p, n, d):
		g = (1 / 2 - p) / np.log(1 / p - 1)
		return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

	def beta_plus(self, p, n, d):
		if p >= 0.5:
			g = p * (1 - p)
		else:
			g = (1 / 2 - p) / np.log(1 / p - 1)
		return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

	# Compute the upper confidence bound, for given estimate and number of sample
	def bound_upper(self, p_est, n, d):
		p_tilde = p_est + np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		up = p_tilde
		down = p_est
		for i in range(16):
			temp = (up + down) / 2
			if (temp - self.beta_minus(temp, n, d)) <= p_est:
				down = temp
			else:
				up = temp
		return (up + down) / 2 - p_est
		
	# Compute the lower confidence bound, for given estimate and number of sample
	def bound_lower(self, p_est, n, d):
		p_tilde = p_est - np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		down = p_tilde
		up = p_est
		for i in range(16):
			temp = (up + down) / 2
			if (temp + self.beta_plus(temp, n, d)) >= p_est:
				up = temp
			else:
				down = temp
		return p_est - (up + down) / 2

	# Tigther than Laplace bounds confidence interval, depending on the estimate value, we have for each transition of each pair 2 bounds
	# The lower (index 0) one and the upper (index 1) one.
	def distances(self, p_estimate, r_estimate):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = self.bound_upper(r_estimate[s, a], n, d)
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s, 0] = self.bound_lower(p_estimate[s, a, next_s], n, d)
					self.p_distances[s, a, next_s, 1] = self.bound_upper(p_estimate[s, a, next_s], n, d)

	# From github RonanFR (code associated with SCAL paper)
	# Compared to the proposition of UCRL2_local here we lower every proba and then put as mass as possible on "good" states instead of putting as
	# mass as possible on the best state before lowering the "bad" states
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > 0) and (l <= self.nS - 1):
			idx = self.nS - 1 - l if not reverse else l
			idx = sorted_indices[idx]
			new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
			max_p[idx] += new_delta
			delta += - new_delta
			l += 1
		return max_p
	
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = self.Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances(p_estimate, r_estimate)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)

# Variant of the previous class where the maximization in EVI is only done on the experimental support and argmax(V)
class UCRL2_Lplus_local3(UCRL2_Lplus):
	def name(self):
		return "UCRL2_Lplus_local3"
	
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8)):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > epsilon) and (l <= self.nS - 1):
			idx = self.nS - 1 - l
			idx = sorted_indices[idx]
			if (l == 0) or self.supports[s, a, idx]:
				new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
				max_p[idx] += new_delta
				delta += - new_delta
			l += 1
		return max_p
	
	def computeSupports(self, p_estimate):
		supports = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for next_s in range(self.nS):
					supports[s, a, next_s] = p_estimate[s, a, next_s] > 0
		return supports
	

	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = self.Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances(p_estimate, r_estimate)
		self.supports = self.computeSupports(p_estimate)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)