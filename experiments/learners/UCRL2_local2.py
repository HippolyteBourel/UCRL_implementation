from learners.UCRL2_local import *
import scipy as sp
import numpy as np

# UCRL modification with component-wise confidance bounds, using the same maximization as SCAL (introduced by Dann and Brunskill)
class UCRL2_local2(UCRL2_local):
	def name(self):
		return "UCRL2_local2"
	
	# From github RonanFR (code associated with SCAL paper)
	# Compared to the proposition of UCRL2_local here we lower every proba and then put as mass as possible on "good" states instead of putting as
	# mass as possible on the best state before lowering the "bad" states
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s]))
			delta += - max_p[next_s]
		l = 0
		while (delta > 0) and (l <= self.nS - 1):
			idx = self.nS - 1 - l if not reverse else l
			idx = sorted_indices[idx]
			new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx] - max_p[idx]))
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
		self.distances()
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)