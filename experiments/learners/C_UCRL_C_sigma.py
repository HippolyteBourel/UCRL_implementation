from learners.C_UCRL_C import *
import scipy as sp
import numpy as np
import copy as cp

# Completed but not tested.

# C_UCRL_C is the C_UCRL(C) algorithm introduced by Maillard and Asadi 2018.
# It extends the UCRL2 class, see this one for commentary about its definition, here only the modifications will be discribed.
# Inputs:
#	nC number of equivalence classes in the MDP
#	C equivalence classes in the MDP, reprensented by a nS x nA matrix C with for each pair (s, a),
#		C[s, a] = c with c natural in  [0, nC - 1] the class of the pair.
class C_UCRL_C_sigma(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC, profile_mapping):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
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
		self.profile_mapping = profile_mapping
	
	def name(self):
		return "C_UCRL(C,mapping)"
	

		# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.vk = np.zeros(self.nC)
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

# Upgraded version using improvements introduced in UCRL_Lplus.UCRL2_Lplus_local3
class C_UCRL_C_sigma_plus(C_UCRL_C14_Lplus_local3):
	def __init__(self,nS, nA, delta, C, nC, profile_mapping):
		self.Pk_c = np.zeros((nC, nS))
		super().__init__(nS, nA, delta, C, nC)
		self.profile_mapping = profile_mapping
	
	def name(self):
		return "C_UCRL(C,mapping) plus"
	
	def distances(self, p_estimate, r_estimate):
		dc = self.delta / (2 * self.nC)
		for s in range(self.nS):
			for a in range(self.nA):
				nc = max(1, self.Nk_c[self.C[s, a]])
				self.r_distances[s, a] = self.bound_upper(r_estimate[s, a], nc, dc)
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s, 0] = self.bound_lower(p_estimate[s, a, next_s], nc, dc)
					self.p_distances[s, a, next_s, 1] = self.bound_upper(p_estimate[s, a, next_s], nc, dc)
	
	# Auxiliary function to compute N the current state-action count.
	def computeN_c(self):
		for c in range(self.nC):
			self.Nk_c[c] = sum([self.Nk[s, a] for (s, a) in self.C_li[c]])
	
	# Auxiliary function to compute R the current accumulated reward.
	def computeR_c(self):
		for c in range(self.nC):
			self.Rk_c[c] = sum([self.Rk[s, a] for (s, a) in self.C_li[c]])
	
	# Auxiliary function to compute P the current transitions count.
	def computeP_c(self):
		P = np.zeros((self.nC, self.nS))
		for t in range(len(self.observations[1])):
			P[self.C[self.observations[0][t], self.observations[1][t]],
				list(self.profile_mapping[self.observations[0][t], self.observations[1][t]]).index(self.observations[0][t+1])] += 1
		self.Pk_c = P
	
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		# First estimate the profile mapping
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.computeR_c()
		self.computeN_c()
		self.computeP_c()
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk_c[self.C[s, a]]])
				r_estimate[s, a] = self.Rk_c[self.C[s, a]] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk_c[self.C[s, a], list(self.profile_mapping[s, a]).index(next_s)] / div
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances(p_estimate, r_estimate)
		#########
		self.supports = self.computeSupports(p_estimate)
		#print(self.t)
		#print(self.Nk)
		#print(p_estimate)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)