from learners.UCRL import *
import scipy as sp
import numpy as np

# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
class UCRL2_local(UCRL2_boost):
	def __init__(self,nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.span = []
	
	def name(self):
		return "UCRL2_local"
	
	#def distances(self, Pk):
	#	d = self.delta / (2 * self.nS * self.nA)
	#	for s in range(self.nS):
	#		for a in range(self.nA):
	#			n = max(1, self.Nk[s, a])
	#			self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
	#			for next_s in range(self.nS):
	#				n = max(1, Pk[s, a, next_s])
	#				self.p_distances[s, a, next_s] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1)) / d) / (2 * n))
	def distances(self):
		Pk = self.Pk
		for s in range(self.nS):
			for a in range(self.nA):
				self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
												/ (2 * max([1, self.Nk[s, a]])))
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
												/ (2 * max([1, Pk[s, a, next_s]])))
					
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a. (modified to deal with local bounds)
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		if reverse:
			sorted_indices = sorted_indices[::-1]
		min1 = min([1, min([p_estimate[s, a, sorted_indices[-1]] + self.p_distances[s, a, sorted_indices[-1]],
							p_estimate[s, a, sorted_indices[-1]] + sum(self.p_distances[s, a]) - self.p_distances[s, a, sorted_indices[-1]]])])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate[s, a])
			max_p[sorted_indices[-1]] = min1
			l = 0
			delta = min1 - p_estimate[s, a, sorted_indices[-1]]
			while (delta > epsilon) and (l < self.nS - 1):
				max_p[sorted_indices[l]] = max([0, max_p[sorted_indices[l]] - min([delta, self.p_distances[s, a, sorted_indices[l]]])])
				delta += (max_p[sorted_indices[l]] - p_estimate[s, a, sorted_indices[l]])
				l += 1
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.argsort(u0)
		itera = 0
		while True:
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1
				self.span.append(max(u1) - min(u1))
				break
			elif itera > max_iter:
				print("No convergence in the EVI at time : ", self.t)
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
				itera += 1
	
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
		self.distances()
		self.EVI(r_estimate, p_estimate)