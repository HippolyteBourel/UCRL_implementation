from learners.C_UCRL import *
import scipy as sp
import numpy as np
import copy as cp

# Work in progress.

# C_UCRL is the algorithm introduced by Maillard and Asadi 2018, it is the more realistic version were C (the classes) and sigma (the profile
# mapping) are unknown. It extends C_UCRL_C which is the implementation of the algorihtm C_UCRL(C) of the paper, using clustering in order to
# estimate C.
class C_UCRL_MSC(C_UCRL):
	def __init__(self,nS, nA, delta, epsilon = 0.01):
		self.nS = nS
		self.nA = nA
		self.nC = nS * nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros(self.nC)
		self.Nk = np.zeros((self.nS, self.nA))
		self.Nk_c = np.zeros(self.nC)
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
		self.C = np.zeros(1)
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.epsilon = epsilon
		self.p_estimate_c = np.zeros((self.nC, self.nS))
		self.Pk_c = np.zeros((self.nC, self.nS))
		self.count = np.zeros(self.nC) # Contains the number of state-action pairs in each class.
		self.deleted = [] # Used during the clustering algorithm to memorize deleted classes.
	
	def name(self):
		return "C_UCRL_MSC"
	
	# Auxiliary function to compute P the current transitions count.
	def computeP_c(self):
		P = np.zeros((self.nC, self.nS))
		for t in range(len(self.observations[1])):
			P[self.C[self.observations[0][t], self.observations[1][t]],
				list(self.profile_mapping[self.observations[0][t], self.observations[1][t]]).index(self.observations[0][t+1])] += 1
		self.Pk_c = P
		return P
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward of C_UCRL and the stopping criterion.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))


	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		# First initialization for the clustering and then running it.
		self.nC = self.nS * self.nA
		self.C = np.arange(self.nC).reshape((self.nS, self.nA))
		self.vk = np.zeros(self.nC)
		self.computeN()
		Rk = self.computeR()
		Pk = self.computeP()
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		r_estimate_c = np.zeros(self.nC)
		p_estimate_c = np.zeros((self.nC, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				c = s * self.nA + a
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = Rk[s, a] / div
				r_estimate_c[c] = Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = Pk[s, a, next_s] / div
					p_estimate_c[c, next_s] = Pk[s, a, next_s] / div
				p_estimate_c[c] = np.sort(p_estimate_c[c]) # We immediately sort it, the "real" transition probability is useless for the
				# clustering and it allows to simplify a lot the code because it is no longer necessary to use the profile mapping in the clustering.
		self.clustering(p_estimate_c, r_estimate_c)
		# Having the clustering we now put it "in good format" (which means redifining C and nC properly). This should be "optimizable" with self.deleted.
		self.nC -= len(self.deleted)
		dico = []
		for c in range(self.nC):
			for s in range(self.nS):
				for a in range(self.nA):
					if not (self.C[s, a] in dico):
						dico.append(self.C[s, a])
					self.C[s, a] = dico.index(self.C[s, a]) # Look very non-optimal to do it that way...
		# And finally we computes the estimates using the previously computed clustering and run the EVI.
		self.computeN_c()
		Rk_c = self.computeR_c()
		Pk_c = self.computeP_c()
		r_estimate_c = np.zeros(self.nC)
		p_estimate_c = np.zeros((self.nC, self.nS))
		for c in range(self.nC):
			div = max([1, self.Nk_c[c]])
			r_estimate_c[c] = Rk_c[c] / div
			for next_s in range(self.nS):
				p_estimate_c[c, next_s] = Pk_c[c, next_s] / div
		self.distances()
		self.deleted = []
		self.update_profile_mapping(p_estimate)
		self.EVI(r_estimate_c, p_estimate_c)
		self.p_estimate_c = p_estimate_c
		self.compute_count()
	
	# To update the list self.count
	def compute_count(self):
		for c in range(self.nC):
			temp = 0
			for li in self.C:
				temp += list(li).count(c)
			self.count[c] = temp
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			self.Pk_c[self.C[s, a], int(list(self.profile_mapping[s, a]).index(state))] += 1
			n = max([1, self.Nk_c[self.C[s, a]] + self.vk[self.C[s, a]]])
			d = self.delta / self.nC
			temp1 = abs(self.p_estimate_c[self.C[s, a], int(list(self.profile_mapping[s, a]).index(state))]
						- (self.Pk_c[self.C[s, a], int(list(self.profile_mapping[s, a]).index(state))] / n))
			temp2 = (1 + self.epsilon) * self.count[self.C[s, a]] * self.bH(n, d)
			if (temp1  > temp2) or (self.vk[self.C[state, action]] >= max([1, self.Nk_c[self.C[state, action]]])): # Stoppping criterion
				self.new_episode()
		action  = self.policy[state]
		return action
