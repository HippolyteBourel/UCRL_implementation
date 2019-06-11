from learners.C_UCRL_C import *
import scipy as sp
import numpy as np
import copy as cp

# Completed but not tested.

# C_UCRL_C is the C_UCRL(C) algorithm introduced by Maillard and Asadi 2018. Here improvement with the Modified Stopping Citerion also introduced by
# Maillard and Asadi 2018.
# It extends the UCRL2 class, see this one for commentary about its definition, here only the modifications will be discribed.
# Inputs:
#	nC number of equivalence classes in the MDP
#	C equivalence classes in the MDP, reprensented by a nS x nA matrix C with for each pair (s, a),
#		C[s, a] = c with c natural in  [0, nC - 1] the class of the pair.
class C_UCRL_C_MSC(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC, epsilon = 0.01):
		print("Initialize C_UCRL_C_MSC with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC = nC
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros(self.nC)
		self.Nk = np.zeros((self.nS, self.nA))
		self.Nk_c = np.zeros(self.nC)
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
		self.C = C
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.epsilon = epsilon
		self.p_estimate_c = np.zeros((self.nC, self.nS))
		self.Pk_c = np.zeros((self.nC, self.nS))
		self.count = np.zeros(self.nC) # Contains the number of state-action pairs in each class.
	
	def name(self):
		return "C_UCRL_C_MSC"
	
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
		self.vk = np.zeros(self.nC)
		# First estimate the profile mapping
		self.computeN()
		Pk = self.computeP()
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = Pk[s, a, next_s] / div
		self.update_profile_mapping(p_estimate)
		# Then initiate the episode as in UCRL2 but using equivalence classes to improve the estimates
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
		

