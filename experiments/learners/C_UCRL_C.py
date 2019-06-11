from learners.UCRL import *
from learners.UCRL2_L import *
from learners.UCRL_Lplus import *
import scipy as sp
import numpy as np
import copy as cp
import itertools as it
import random as rd
import math
import pylab as pl

# C_UCRL_C is the C_UCRL(C) algorithm introduced by Maillard and Asadi 2018.
# It extends the UCRL2 class, see this one for commentary about its definition, here only the modifications will be discribed.
# Inputs:
#	nC number of equivalence classes in the MDP
#	C equivalence classes in the MDP, reprensented by a nS x nA matrix C with for each pair (s, a),
#		C[s, a] = c with c natural in  [0, nC - 1] the class of the pair.
class C_UCRL_C(UCRL2):
	def __init__(self,nS, nA, delta, C, nC):
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
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.u = np.zeros(self.nS)

	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])
	
	def name(self):
		return "C_UCRL_C"

	# Auxiliary function to compute N the current state-action count.
	def computeN_c(self):
		N = np.zeros(self.nC)
		for t in range(len(self.observations[1])):
			N[self.C[self.observations[0][t], self.observations[1][t]]] += 1
		self.Nk_c = N
	
	# Auxiliary function to compute R the current accumulated reward.
	def computeR_c(self):
		R = np.zeros(self.nC)
		for t in range(len(self.observations[1])):
			R[self.C[self.observations[0][t], self.observations[1][t]]] += self.observations[2][t]
		self.Rk = R
		return R
	
	# Auxiliary function to compute P the current transitions count.
	def computeP_c(self):
		P = np.zeros((self.nC, self.nS))
		for t in range(len(self.observations[1])):
			P[self.C[self.observations[0][t], self.observations[1][t]],
				list(self.profile_mapping[self.observations[0][t], self.observations[1][t]]).index(self.observations[0][t+1])] += 1
		self.Pk = P
		return P
	
	# Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	# Tighter bounds are used in C_UCRL.
	def distances(self):
		d = self.delta / (2 * self.nC)
		for c in range(self.nC):
			n = max(1, self.Nk_c[c])
			self.r_distances[c] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
			self.p_distances[c] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# p_estimate CxS ?
	def max_proba(self, p_estimate, sorted_indices, c, s, a):
		min1 = min([1, p_estimate[c, int(list(self.profile_mapping[s, a]).index(sorted_indices[-1]))] + (self.p_distances[c] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			#max_p = [p_estimate[c, int(i)] for i in self.profile_mapping[s, a]]
			for ss in range(self.nS):
				max_p[ss] = p_estimate[c, list(self.profile_mapping[s, a]).index(ss)]
			max_p[sorted_indices[-1]] += self.p_distances[c] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		current_iter = 0
		u0 = np.zeros(self.nS)
		if self.t > 1:
			u0 = self.u
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		while True:
			current_iter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					c = self.C[s, a]
					max_p = self.max_proba(p_estimate, sorted_indices, c, s, a)
					temp = min((1, r_estimate[c] + self.r_distances[c])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if current_iter > max_iter:
				print("No convergence in the EVI")
				self.u = u1
				break
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)

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
		#print(p_estimate)
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

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.u = np.zeros(self.nS)
		self.observations = [[inistate], [], []]
		self.new_episode()
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		#if self.t == 1:
		#	self.to_plot  =[]
		#else:
		#	self.to_plot.append(self.nC)
		#if self.t == 100000:
		#	pl.figure()
		#	pl.plot(self.to_plot)
		#	pl.show()
		action = self.policy[state]
		if self.vk[self.C[state, action]] >= max([1, self.Nk_c[self.C[state, action]]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		#if (self.t % 1000) == 0:
			#print("At time : ", self.t, "\nnC = ", self.nC, "\nAnd C = =", self.C)
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[self.C[state, action]] += 1
		self.t += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)





# The idea here is to randomly (with a decreasing probability over the time) replace the estimated profile by a l-shift permutation (with l increasing)
# it time we use the shift to, over the time, test all l-shifts.
class C_UCRL_C2(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC):
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
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.shift = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C2"
	
	def computeShift(self, s, a):
		l = self.shift[s, a]
		res = np.array([((i + l) % self.nS) for i in range(self.nS)])
		self.shift[s, a] += 1
		return res
	
	def update_profile_mapping(self, p):
		K = 100
		for s in range(self.nS):
			for a in range(self.nA):
				epsilon = K / max((1, self.Nk[s, a]))
				if rd.random() < epsilon:
					self.profile_mapping[s, a] = self.computeShift(s, a)
				else:
					self.profile_mapping[s, a] = np.argsort(p[s, a])

	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.u = np.zeros(self.nS)
		self.observations = [[inistate], [], []]
		self.shift = np.zeros((self.nS, self.nA))
		self.new_episode()



class C_UCRL_C2_sqrtSC(C_UCRL_C2):
	def name(self):
		return "C_UCRL_C2_sqrtSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[self.C[state, action]] >= np.sqrt(max([1, self.Nk_c[self.C[state, action]]])): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		#if (self.t % 1000) == 0:
			#print("At time : ", self.t, "\nnC = ", self.nC, "\nAnd C = =", self.C)
		return action


class C_UCRL_C2bis_sqrtSC(C_UCRL_C2_sqrtSC):
	def name(self):
		return "C_UCRL_C2bis_sqrtSC"
	
	def update_profile_mapping(self, p):
		K = 100
		for s in range(self.nS):
			for a in range(self.nA):
				epsilon = K / self.t
				if rd.random() < epsilon:
					self.profile_mapping[s, a] = self.computeShift(s, a)
				else:
					self.profile_mapping[s, a] = np.argsort(p[s, a])


class C_UCRL_C2bis(C_UCRL_C2):
	def name(self):
		return "C_UCRL_C2bis"
	
	def update_profile_mapping(self, p):
		K = 100
		for s in range(self.nS):
			for a in range(self.nA):
				epsilon = K / self.t
				if rd.random() < epsilon:
					self.profile_mapping[s, a] = self.computeShift(s, a)
				else:
					self.profile_mapping[s, a] = np.argsort(p[s, a])







# The idea here is to exclude from the classes the unsampled state-action pairs (implemented having the idea of excluding from classes all states-actions
# pairs sampled less than N0 times with arbitrary N0 not necesseraly equal to 1 for further tests)
class C_UCRL_C3(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C3"
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		for s in range(self.nS):
			for a in range(self.nA):
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					self.C[s, a] = self.nC
					self.nC += 1
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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



# Fusion of C_UCRL_C2bis and C_UCRL_C3
class C_UCRL_C4(C_UCRL_C2bis):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.shift = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C4"
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		for s in range(self.nS):
			for a in range(self.nA):
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					self.C[s, a] = self.nC
					self.nC += 1
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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




# The idea here is to add the "alpha" criterion of C_UCRL in C_UCRL_C (this criterion is the one used in div4C and div2C algorithms, in practice the
# only version of C_UCRL really working on all our experiments so far). The idea behind this is to put together the element with a good approximation
# of the profile mapping (and to exclude the other, wich should increase the probability to visit and so sample them) (version with alpha fixed)
class C_UCRL_C5_fixed(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC, alpha = 4):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.alpha = alpha
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C5_fixed"
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		sub_classes = [[] for _ in range(self.nC_true)]
		N_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				c_true = self.C_true[s, a]
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					if sub_classes[c_true] == []:
						sub_classes[c_true].append(c_true)
						N_sub[c_true] = 0
					else:
						c = self.nC
						sub_classes[c_true].append(c)
						N_sub[c] = 0
						self.nC += 1
						self.C[s, a] = c
				else:
					todo = True
					for c in sub_classes[c_true]:
						if (N_sub[c] <= self.alpha * self.Nk[s, a]) and (self.Nk[s, a] <= self.alpha * N_sub[c]):
							self.C[s, a] = c
							todo = False
							break
					if todo:
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							N_sub[c_true] = self.Nk[s, a]
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							N_sub[c] = self.Nk[s, a]
							self.nC += 1
							self.C[s, a] = c
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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



# (increasing alpha version)
class C_UCRL_C5_increasing(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.alpha = 1
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C5_increasing"
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		
		# Updating alpha
		self.alpha = 1 + np.log(self.t)
		
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		sub_classes = [[] for _ in range(self.nC_true)]
		N_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				c_true = self.C_true[s, a]
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					if sub_classes[c_true] == []:
						sub_classes[c_true].append(c_true)
						N_sub[c_true] = 0
					else:
						c = self.nC
						sub_classes[c_true].append(c)
						N_sub[c] = 0
						self.nC += 1
						self.C[s, a] = c
				else:
					todo = True
					for c in sub_classes[c_true]:
						if (N_sub[c] <= self.alpha * self.Nk[s, a]) and (self.Nk[s, a] <= self.alpha * N_sub[c]):
							self.C[s, a] = c
							todo = False
							break
					if todo:
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							N_sub[c_true] = self.Nk[s, a]
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							N_sub[c] = self.Nk[s, a]
							self.nC += 1
							self.C[s, a] = c
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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



# The idea here is to put in the same class only elements with same support in the estimate of the transition (and non nul number samples)
class C_UCRL_C6(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.u = np.zeros(self.nS)
	
	def name(self):
		return "C_UCRL_C6"
	
	def computeSupports(self, p_estimate):
		supports = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				supports[s, a] = sum([(p_estimate[s, a, ss] > 0) for ss in range(self.nS)])
		return supports
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a) and cluster by support of the transition
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		supports = self.computeSupports(p_estimate)
		sub_classes = [[] for _ in range(self.nC_true)]
		supports_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				c_true = self.C_true[s, a]
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					if sub_classes[c_true] == []:
						sub_classes[c_true].append(c_true)
					else:
						c = self.nC
						sub_classes[c_true].append(c)
						self.nC += 1
						self.C[s, a] = c
				else:
					todo = True
					for c in sub_classes[c_true]:
						if supports_sub[c] == supports[s, a]:
							self.C[s, a] = c
							todo = False
							break
					if todo:
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							supports_sub[c_true] = supports[s, a]
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							supports_sub[c] = supports[s, a]
							self.nC += 1
							self.C[s, a] = c
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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


# The idea of this version is to keep the alpha criterion intoduced in the C_UCRL_C5 classes (and which is effecient in experiments, it
# should theoretically bound, or at least reduce, the bias produced by the unknown sigma) and we had an optimistic choice of sigma in the
# EVI, building the set of possible sigma by looking at the intersection of confidence bound around the estimates of transtion.
# Finally we stop taking care of the alpha criterion when the set of plausible sigma is reduced to one element (because at this point
# there's no longer any inconsitency/bias brougth by the unknown sigma)
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# Currently when we ignore alpha we arbitrariliy put (s, a) in the same subclass as the the first (in lexicographique order) (s, a) of the class,
# it not good to this and HAVE TO BE CHANGED
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
### To take off the gestion of experimental bounds: take off: not (bounds[s, a, ss] == 0 or bounds[s, a, p_ss] == 0) in computesigmaset
#### and take off: if Pk[s, a, ss] == 0:    bounds[s, a, ss] = 0 in computeBounds (it should be enough)
class C_UCRL_C7_fixed(C_UCRL_C5_fixed):
	def __init__(self,nS, nA, delta, C, nC, alpha = 4, T = 100000):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.alpha = alpha
		self.u = np.zeros(self.nS)
		self.SigmaSet = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for t in range(1, T):
			temp = 2 * self.nS**2 * (1 + math.ceil(np.log(2*T / t) / np.log(4/3))) * np.exp(- t)
			if temp <= self.delta:
				self.tau = t
				break
	
	def name(self):
		return "C_UCRL_C7_fixed"
	
	# compute the set of plausible sigma looking at the incertitude around the transition probability (on which the estimated sigma is built)
	def computeSigmaSet(self, p_estimate, bounds):
		self.SigmaSet = np.zeros((self.nS, self.nA, self.nS, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS - 1):
					for p_ss in range(ss + 1, self.nS): # does not compare ss and p_ss (because its not a "new" sigma)
						test1 = (p_estimate[s, a, ss] + bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test2 = (p_estimate[s, a, ss] + bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						test3 = (p_estimate[s, a, ss] - bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test4 = (p_estimate[s, a, ss] - bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						case1 = test1 and test2
						case2 = test3 and test4
						case3 = not (test2 and test3)
						if (case1 or case2 or case3) and not (bounds[s, a, ss] == 0 or bounds[s, a, p_ss] == 0):
							self.SigmaSet[s, a, ss, p_ss] = self.SigmaSet[s, a, p_ss, ss] = 1
	
	# return the bounds used on the transition probability to build the set of plausible sigma
	def computeBounds(self, p_estimate, Pk):
		l = (4 / 3) * self.tau
		bounds = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				n = self.Nk[s, a]
				for ss in range(self.nS):
					if Pk[s, a, ss] == 0:
						bounds[s, a, ss] = 0
					else:
						bounds[s, a, ss] = (np.sqrt(l / (2 * n)) + np.sqrt(((3 * l) / (2 *n))
										+ np.sqrt((2 * p_estimate[s, a, ss] * (1 - p_estimate[s, a, ss]) * l) / n)))**2
		return bounds
	
	# return an S x A matrice with a zero when there's only one plausible permutation for (s, a) and 1 otherwise (used to cancel the alpha crtierion
	# in the first case)
	def computeSetNoAlpha(self):
		test = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS):
					if 1. in self.SigmaSet[s, a, ss]:
						test[s, a] = 1
		return test

	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#########
		bounds = self.computeBounds(p_estimate, Pk)
		self.computeSigmaSet(p_estimate, bounds) # used in max_proba for the optimism on the profile mapping
		test_noalpha = self.computeSetNoAlpha()
		#print(self.SigmaSet)
		#########
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		sub_classes = [[] for _ in range(self.nC_true)]
		N_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				if test_noalpha[s, a] == 1:
					c_true = self.C_true[s, a]
					if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							N_sub[c_true] = 0
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							N_sub[c] = 0
							self.nC += 1
							self.C[s, a] = c
					else:
						todo = True
						for c in sub_classes[c_true]:
							if (N_sub[c] <= self.alpha * self.Nk[s, a]) and (self.Nk[s, a] <= self.alpha * N_sub[c]):
								self.C[s, a] = c
								todo = False
								break
						if todo:
							if sub_classes[c_true] == []:
								sub_classes[c_true].append(c_true)
								N_sub[c_true] = self.Nk[s, a]
							else:
								c = self.nC
								sub_classes[c_true].append(c)
								N_sub[c] = self.Nk[s, a]
								self.nC += 1
								self.C[s, a] = c
				#else:
				#	print("No alpha for ", s, a, " at time : ", self.t)
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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
	
	
	# Computing the argmax plausible sigma in the Extended Value Iteration for given state s and action a.
	def max_sigma(self, sorted_indices, s, a):
		max_sigma = np.array(cp.deepcopy(self.profile_mapping[s, a]))
		for i in range(self.nS - 1, list(max_sigma).index(sorted_indices[-1]), -1):
			if self.SigmaSet[s, a, sorted_indices[-1], int(max_sigma[i])] == 1:
				max_sigma[list(max_sigma).index(sorted_indices[-1])] , max_sigma[i] = max_sigma[i], max_sigma[list(max_sigma).index(sorted_indices[-1])]
				break
		for l in range(self.nS - 1):
			for i in range(list(max_sigma).index(sorted_indices[l])):
				if ((self.SigmaSet[s, a, sorted_indices[l], int(max_sigma[i])] == 1) and (l < list(sorted_indices).index(max_sigma[i]))
					and (self.SigmaSet[s, a, sorted_indices[l], int(self.profile_mapping[s, a, i])] == 1)):
					max_sigma[list(max_sigma).index(sorted_indices[l])] , max_sigma[i] = max_sigma[i], max_sigma[list(max_sigma).index(sorted_indices[l])]
					break
		return max_sigma

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, c, s, a):
		min1 = min([1, p_estimate[c, int(list(self.sigma[s, a]).index(sorted_indices[-1]))] + (self.p_distances[c] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			#max_p = [p_estimate[c, int(i)] for i in self.profile_mapping[s, a]]
			for ss in range(self.nS):
				max_p[ss] = p_estimate[c, list(self.sigma[s, a]).index(ss)]
			max_p[sorted_indices[-1]] += self.p_distances[c] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 100):
		current_iter = 0
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		self.sigma = np.zeros((self.nS, self.nA, self.nS))
		while True:
			current_iter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					c = self.C[s, a]
					self.sigma[s, a] = self.max_sigma(sorted_indices, s, a)
					max_p = self.max_proba(p_estimate, sorted_indices, c, s, a)
					temp = min((1, r_estimate[c] + self.r_distances[c])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if current_iter > max_iter:
				print("No convergence in the EVI")
				break
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)











# The idea of this version is to keep the alpha criterion intoduced in the C_UCRL_C5 classes (and which is effecient in experiments, it
# should theoretically bound, or at least reduce, the bias produced by the unknown sigma) and we had an optimistic choice of sigma in the
# EVI, building the set of possible sigma by looking at the intersection of confidence bound around the estimates of transtion.
# Finally we stop taking care of the alpha criterion when the set of plausible sigma is reduced to one element (because at this point
# there's no longer any inconsitency/bias brougth by the unknown sigma)
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# Currently when we ignore alpha we arbitrariliy put (s, a) in the same subclass as the the first (in lexicographique order) (s, a) of the class,
# it not good to this and HAVE TO BE CHANGED
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
### To take off the gestion of experimental bounds: take off: not (bounds[s, a, ss] == 0 or bounds[s, a, p_ss] == 0) in computesigmaset
#### and take off: if Pk[s, a, ss] == 0:    bounds[s, a, ss] = 0 in computeBounds (it should be enough) <- done in this version (not in 7)
class C_UCRL_C8_fixed(C_UCRL_C5_fixed):
	def __init__(self,nS, nA, delta, C, nC, alpha = 4, T = 100000):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.alpha = alpha
		self.u = np.zeros(self.nS)
		self.SigmaSet = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for t in range(1, T):
			temp = 2 * self.nS**2 * (1 + math.ceil(np.log(2*T / t) / np.log(4/3))) * np.exp(- t)
			if temp <= self.delta:
				self.tau = t
				break
	
	def name(self):
		return "C_UCRL_C8_fixed"
	
	# compute the set of plausible sigma looking at the incertitude around the transition probability (on which the estimated sigma is built)
	def computeSigmaSet(self, p_estimate, bounds):
		self.SigmaSet = np.zeros((self.nS, self.nA, self.nS, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS - 1):
					for p_ss in range(ss + 1, self.nS): # does not compare ss and p_ss (because its not a "new" sigma)
						test1 = (p_estimate[s, a, ss] + bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test2 = (p_estimate[s, a, ss] + bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						test3 = (p_estimate[s, a, ss] - bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test4 = (p_estimate[s, a, ss] - bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						case1 = test1 and test2
						case2 = test3 and test4
						case3 = not (test2 and test3)
						if (case1 or case2 or case3):
							self.SigmaSet[s, a, ss, p_ss] = self.SigmaSet[s, a, p_ss, ss] = 1
	
	# return the bounds used on the transition probability to build the set of plausible sigma
	def computeBounds(self, p_estimate, Pk):
		l = (4 / 3) * self.tau
		bounds = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				n = max((1, self.Nk[s, a]))
				for ss in range(self.nS):
					bounds[s, a, ss] = (np.sqrt(l / (2 * n)) + np.sqrt(((3 * l) / (2 *n))
										+ np.sqrt((2 * p_estimate[s, a, ss] * (1 - p_estimate[s, a, ss]) * l) / n)))**2
		return bounds
	
	# return an S x A matrice with a zero when there's only one plausible permutation for (s, a) and 1 otherwise (used to cancel the alpha crtierion
	# in the first case)
	def computeSetNoAlpha(self):
		test = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS):
					if 1. in self.SigmaSet[s, a, ss]:
						test[s, a] = 1
		return test

	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#########
		bounds = self.computeBounds(p_estimate, Pk)
		self.computeSigmaSet(p_estimate, bounds) # used in max_proba for the optimism on the profile mapping
		test_noalpha = self.computeSetNoAlpha()
		#print(self.SigmaSet)
		#########
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		sub_classes = [[] for _ in range(self.nC_true)]
		N_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				if False:#test_noalpha[s, a] == 1:
					c_true = self.C_true[s, a]
					if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							N_sub[c_true] = 0
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							N_sub[c] = 0
							self.nC += 1
							self.C[s, a] = c
					else:
						todo = True
						for c in sub_classes[c_true]:
							if (N_sub[c] <= self.alpha * self.Nk[s, a]) and (self.Nk[s, a] <= self.alpha * N_sub[c]):
								self.C[s, a] = c
								todo = False
								break
						if todo:
							if sub_classes[c_true] == []:
								sub_classes[c_true].append(c_true)
								N_sub[c_true] = self.Nk[s, a]
							else:
								c = self.nC
								sub_classes[c_true].append(c)
								N_sub[c] = self.Nk[s, a]
								self.nC += 1
								self.C[s, a] = c
				#else:
				#	print("No alpha for ", s, a, " at time : ", self.t)
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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
	
	
	# Computing the argmax plausible sigma in the Extended Value Iteration for given state s and action a.
	def max_sigma(self, sorted_indices, s, a):
		max_sigma = np.array(cp.deepcopy(self.profile_mapping[s, a]))
		for i in range(self.nS - 1, list(max_sigma).index(sorted_indices[-1]), -1):
			if self.SigmaSet[s, a, sorted_indices[-1], int(max_sigma[i])] == 1:
				max_sigma[list(max_sigma).index(sorted_indices[-1])] , max_sigma[i] = max_sigma[i], max_sigma[list(max_sigma).index(sorted_indices[-1])]
				break
		for l in range(self.nS - 1):
			for i in range(list(max_sigma).index(sorted_indices[l])):
				if ((self.SigmaSet[s, a, sorted_indices[l], int(max_sigma[i])] == 1) and (l < list(sorted_indices).index(max_sigma[i]))
					and (self.SigmaSet[s, a, sorted_indices[l], int(self.profile_mapping[s, a, i])] == 1)):
					max_sigma[list(max_sigma).index(sorted_indices[l])] , max_sigma[i] = max_sigma[i], max_sigma[list(max_sigma).index(sorted_indices[l])]
					break
		return max_sigma

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, c, s, a):
		min1 = min([1, p_estimate[c, int(list(self.sigma[s, a]).index(sorted_indices[-1]))] + (self.p_distances[c] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			#max_p = [p_estimate[c, int(i)] for i in self.profile_mapping[s, a]]
			for ss in range(self.nS):
				max_p[ss] = p_estimate[c, list(self.sigma[s, a]).index(ss)]
			max_p[sorted_indices[-1]] += self.p_distances[c] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 100):
		current_iter = 0
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		self.sigma = np.zeros((self.nS, self.nA, self.nS))
		while True:
			current_iter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					c = self.C[s, a]
					self.sigma[s, a] = self.max_sigma(sorted_indices, s, a)
					max_p = self.max_proba(p_estimate, sorted_indices, c, s, a)
					temp = min((1, r_estimate[c] + self.r_distances[c])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if current_iter > max_iter:
				print("No convergence in the EVI")
				break
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)







# Alpha fixed (as C_UCRL_C5_fixed) and "true?" confidence bounds (no longer Nt_c dependant, because wrong when sigma is unknown)
class C_UCRL_C9_fixed(C_UCRL_C5_fixed):
	def name(self):
		return "C_UCRL_C9_fixed"
	
	# Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	# Tighter bounds are used in C_UCRL.
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = self.Nk[s, a]
				if n > 0:
					self.r_distances[self.C[s, a]] += (np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
													    ) * (self.Nk[s, a] / self.Nk_c[self.C[s, a]])
					self.p_distances[self.C[s, a]] += (np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
														) * (self.Nk[s, a] / self.Nk_c[self.C[s, a]])
				else:
					n = 1
					self.r_distances[self.C[s, a]] += np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					self.p_distances[self.C[s, a]] += np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)


# Original C_UCRL_C but with the "true?" confidence bounds on the estimates.
class C_UCRL_C9(C_UCRL_C):
	def name(self):
		return "C_UCRL_C9"
	
	# Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	# Tighter bounds are used in C_UCRL.
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = self.Nk[s, a]
				if n > 0:
					self.r_distances[self.C[s, a]] += (np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
													    ) * (self.Nk[s, a] / self.Nk_c[self.C[s, a]])
					self.p_distances[self.C[s, a]] += (np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
														) * (self.Nk[s, a] / self.Nk_c[self.C[s, a]])
				else:
					n = 1
					self.r_distances[self.C[s, a]] += np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					self.p_distances[self.C[s, a]] += np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)



# The idea here is to use the previous bounds (the real ones on q not p..), but instead of doing subclasses (and same estimate for all elements in the class)
# we only use samples from other elements having more samples than us, then looking at these previous bounds we only imporve confidene bounds
# but in practice we don't have classes
class C_UCRL_C10(UCRL2_L_boost):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.C_li = [[] for _ in range(nC)]
		for s in range(nS):
			for a in range(nA):
				self.C_li[C[s, a]].append((s, a))
		self.C = C
		self.nC = nC
		self.n = 10
		self.profile_mapping = np.zeros((nS, nA, nS))
		self.C_sorted = []
		super().__init__(nS, nA, delta)
	
	def name(self):
		return "C_UCRL_C10"
	
	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])

	def computeSortedC(self):
		self.C_sorted = []
		for c in range(self.nC):
			li = self.C_li[c]
			temp = np.argsort([self.Nk[s, a] for (s, a) in li])
			self.C_sorted.append([li[i] for i in temp[::-1]])
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.computeSortedC()
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate0 = np.zeros((self.nS, self.nA, self.nS))
		r_estimate0 = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate0[s, a] = self.Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate0[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.update_profile_mapping(p_estimate0)
		# Then initiate the episode as in UCRL2 but using equivalence classes to improve the estimatee
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.distances()
		r_distances0 = cp.deepcopy(self.r_distances)
		p_distances0 = cp.deepcopy(self.p_distances)
		# TO FINISH
		for c in range(self.nC):
			div0 = 0
			for i in range(len(self.C_sorted[c])):
				(s, a) = self.C_sorted[c][i]
				div0 += self.Nk[s, a]
				div = max((1, div0))
				if (i == 0) or (self.Nk[s, a] <= self.n):
					r_estimate[s, a] = r_estimate0[s, a]
					p_estimate[s, a] = p_estimate0[s, a]
				else:
					r_estimate[s, a] = sum([r_estimate0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					self.r_distances[s, a] = sum([r_distances0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					self.p_distances[s, a] = sum([p_distances0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					for ss in range(self.nS):
						p_estimate[s, a, int(self.profile_mapping[s, a, ss])] = sum([p_estimate0[s2, aa, int(self.profile_mapping[s2, aa, ss])] *
																				self.Nk[s2, aa] for (s2, aa) in self.C_sorted[c][: (i + 1)]]) / div
		#print(self.Nk)
		self.EVI(r_estimate0, p_estimate0)



# Here we keep the same idea as C_UCRL_C10 but add some kind of optimism on the profile mapping in the extended value iteration, precisely
# we optimistically place the transition probability mass over the transitions whicha re out from the experimental support.
class C_UCRL_C11(C_UCRL_C10):
	def name(self):
		return "C_UCRL_C11"
	
	# return an nSxnA matrix of lists with in the list the states s' such that no transition from s doing a to s' has been observed
	# with (s, a) the position of the list in the matrix.
	def computeSigmaSupport(self, p_estimate0, tau = 10**(-8)):
		self.SigmaSupport = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nA):
					if p_estimate0[s, a, ss] < tau:
						self.SigmaSupport[s][a].append(ss)


	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.#MODIFY (replace p_estimate bu p_maxsigma)
	def max_proba(self, p_estimate, sorted_indices, s, a):
		#p_estimate = p_estimate[s, a]
		min1 = min([1, p_estimate[sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate)
			max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p

	# Optimism on the profile mapping for the transitions outside from the experimental support
	def max_sigma(self, p_estimate, u, s, a, tau = 10**(-8)):
		p_max = cp.deepcopy(p_estimate[s, a])
		mass = sum([p_max[i] for i in self.SigmaSupport[s][a]])
		if mass > tau:
			maxV = max([u[i] for i in self.SigmaSupport[s][a]])
			argmax = []
			for i in self.SigmaSupport[s][a]:
				if u[i] >= maxV - tau:
					argmax.append(i)
			n = len(argmax)
			for i in self.SigmaSupport[s][a]:
				if i in argmax:
					p_max[i] = mass / n
				else:
					p_max[i] = 0
		return p_max
				

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.# MODIFY
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		niter = 0
		while True:
			niter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					p_max = self.max_sigma(p_estimate, u0, s, a)
					max_p = self.max_proba(p_max, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0


	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.computeSortedC()
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate0 = np.zeros((self.nS, self.nA, self.nS))
		r_estimate0 = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate0[s, a] = self.Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate0[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.update_profile_mapping(p_estimate0)
		# Then initiate the episode as in UCRL2 but using equivalence classes to improve the estimatee
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.distances()
		r_distances0 = cp.deepcopy(self.r_distances)
		p_distances0 = cp.deepcopy(self.p_distances)
		for c in range(self.nC):
			div0 = 0
			for i in range(len(self.C_sorted[c])):
				(s, a) = self.C_sorted[c][i]
				div0 += self.Nk[s, a]
				div = max((1, div0))
				if (i == 0) or (self.Nk[s, a] <= self.n):
					r_estimate[s, a] = r_estimate0[s, a]
					p_estimate[s, a] = p_estimate0[s, a]
				else:
					r_estimate[s, a] = sum([r_estimate0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					self.r_distances[s, a] = sum([r_distances0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					self.p_distances[s, a] = sum([p_distances0[ss, aa] * self.Nk[ss, aa] for (ss, aa) in self.C_sorted[c][: (i + 1)]]) / div
					for ss in range(self.nS):
						p_estimate[s, a, int(self.profile_mapping[s, a, ss])] = sum([p_estimate0[s2, aa, int(self.profile_mapping[s2, aa, ss])] *
																				self.Nk[s2, aa] for (s2, aa) in self.C_sorted[c][: (i + 1)]]) / div
		#print(self.Nk)
		self.computeSigmaSupport(p_estimate0)
		self.EVI(r_estimate, p_estimate)


# Here we combine the alpha sub-classes (C5_fixed), the bounds used since C9_fixed and the optimism in sigma for elements that are not in the
# experimental support (as C11).
class C_UCRL_C12(C_UCRL_C9_fixed): # modify EVI and maxproba, add max_sigma
	def name(self):
		return "C_UCRL_C12"
	
	# return an nSxnA matrix of lists with in the list the states s' such that no transition from s doing a to s' has been observed
	# with (s, a) the position of the list in the matrix.
	def computeSigmaSupport(self, p_estimate0, tau = 10**(-8)):
		self.SigmaSupport = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nA):
					if p_estimate0[s, a, ss] < tau:
						self.SigmaSupport[s][a].append(ss)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# p_estimate CxS ?
	def max_proba(self, p_max, sorted_indices, c, s, a):
		min1 = min([1, p_max[int(list(self.profile_mapping[s, a]).index(sorted_indices[-1]))] + (self.p_distances[c] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			#max_p = [p_estimate[c, int(i)] for i in self.profile_mapping[s, a]]
			for ss in range(self.nS):
				max_p[ss] = p_max[list(self.profile_mapping[s, a]).index(ss)]
			max_p[sorted_indices[-1]] += self.p_distances[c] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p
	
	# Optimism on the profile mapping for the transitions outside from the experimental support
	def max_sigma(self, p_estimate, u, s, a, c, tau = 10**(-8)):
		p_max = cp.deepcopy(p_estimate[c])
		mass = sum([p_max[list(self.profile_mapping[s, a]).index(i)] for i in self.SigmaSupport[s][a]])
		if mass > tau:
			maxV = max([u[i] for i in self.SigmaSupport[s][a]])
			argmax = []
			for i in self.SigmaSupport[s][a]:
				if u[i] >= maxV - tau:
					argmax.append(i)
			n = len(argmax)
			for i in self.SigmaSupport[s][a]:
				if i in argmax:
					p_max[list(self.profile_mapping[s, a]).index(i)] = mass / n
				else:
					p_max[list(self.profile_mapping[s, a]).index(i)] = 0
		return p_max

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		current_iter = 0
		u0 = np.zeros(self.nS)
		if self.t > 1:
			u0 = self.u
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		while True:
			current_iter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					c = self.C[s, a]
					p_max = self.max_sigma(p_estimate, u0, s, a, c)
					max_p = self.max_proba(p_max, sorted_indices, c, s, a)
					temp = min((1, r_estimate[c] + self.r_distances[c])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if current_iter > max_iter:
				print("No convergence in the EVI")
				self.u = u1
				break
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
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
		#print(p_estimate)
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.nC = self.nC_true
		self.C = cp.deepcopy(self.C_true)
		sub_classes = [[] for _ in range(self.nC_true)]
		N_sub = np.zeros(self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				c_true = self.C_true[s, a]
				if self.Nk[s, a] < 1: # just modify here to ask an arbitrary number of sample
					if sub_classes[c_true] == []:
						sub_classes[c_true].append(c_true)
						N_sub[c_true] = 0
					else:
						c = self.nC
						sub_classes[c_true].append(c)
						N_sub[c] = 0
						self.nC += 1
						self.C[s, a] = c
				else:
					todo = True
					for c in sub_classes[c_true]:
						if (N_sub[c] <= self.alpha * self.Nk[s, a]) and (self.Nk[s, a] <= self.alpha * N_sub[c]):
							self.C[s, a] = c
							todo = False
							break
					if todo:
						if sub_classes[c_true] == []:
							sub_classes[c_true].append(c_true)
							N_sub[c_true] = self.Nk[s, a]
						else:
							c = self.nC
							sub_classes[c_true].append(c)
							N_sub[c] = self.Nk[s, a]
							self.nC += 1
							self.C[s, a] = c
		self.vk = np.zeros(self.nC)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
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
		self.computeSigmaSupport(p_estimate)
		self.EVI(r_estimate_c, p_estimate_c)



# Simplified version, here we use the classes only when sigma is know with high probability
# We add as additional input the size of the support of the transition function for each pair (s, a)
# Remark about compututional time: As in practice this algorithm will not aggregate anything at the beginning (and maybe not a lot later on) we could
# have a much more efficient implementation by filtering the aggregation as the exception instead of trying at each episode to see classes
# (compute sigma etc..)
class C_UCRL_C13(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC, alpha = 4, T = 100000, sizeSupport = None):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nS = nS
		self.nA = nA
		self.nC_true = nC
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
		self.C_true = C
		self.C = np.zeros((self.nS, self.nA))
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.alpha = alpha
		self.u = np.zeros(self.nS)
		self.SigmaTest = np.zeros((self.nS, self.nA))
		#for t in range(1, T):
		#	temp = self.nS * (1 + math.ceil(np.log(2*T / t) / np.log(1.1))) * np.exp(- t)
		#	if temp <= self.delta:
		#		self.tau = t # parameter of the element wise confidence bounds defined in computeBounds function
		#		break
		#self.tau = np.log(math.ceil((1 / delta) * (np.log())))
		self.sizeSupport = sizeSupport # SxA matrix with the size of the support of the transition function for each (s, a)
		self.support = np.zeros((self.nS, self.nA)) # sizes of current experimental support
		self.C_li = [[] for _ in range(self.nC)]
		for s in range(self.nS):
			for a in range(self.nA):
				self.C_li[C[s, a]].append((s, a))
	
	def name(self):
		return "C_UCRL_C13"
	
	# Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	# Tighter bounds are used in C_UCRL.
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS): # Non optimal to do it for each (s, a) but shortcut to have c and c_true..
			for a in range(self.nA):
				n_true = max(1, self.Nk_c_true[self.C_true[s, a]])
				self.r_distances[self.C[s, a]] = np.sqrt(((1 + 1 / n_true) * np.log(2 * np.sqrt(n_true + 1) / d)) / (2 * n_true))
				n = max(1, self.Nk_c[self.C[s, a]])
				self.p_distances[self.C[s, a]] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)

	# for each pair (s, a) comput if on the experimental support there exist two states that could be not well ordered in experimental profile mapping
	# looking at the confidence bounds arounds the transitions probability
	def computeSigmaTest(self, p_estimate, bounds):
		self.SigmaTest = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS - 1):
					for p_ss in range(ss + 1, self.nS): # does not compare ss and p_ss (because its not a "new" sigma)
						test1 = (p_estimate[s, a, ss] + bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test2 = (p_estimate[s, a, ss] + bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						test3 = (p_estimate[s, a, ss] - bounds[s, a, ss] >= p_estimate[s, a, p_ss] - bounds[s, a, p_ss])
						test4 = (p_estimate[s, a, ss] - bounds[s, a, ss] <= p_estimate[s, a, p_ss] + bounds[s, a, p_ss])
						case1 = test1 and test2
						case2 = test3 and test4
						case3 = not (test2 and test3)
						if (case1 or case2 or case3) and not (bounds[s, a, ss] == 0 or bounds[s, a, p_ss] == 0):
							self.SigmaTest[s, a] = 1
							break
	
	# return element-wise bounds on transition probability, these confidence bounds are used to check if there is multiple plausible profile
	# mapping or not (and in this case we aggregate)
	# Also compute the size of the experimental support of the transition function for all pair (s, a) (in the variable self.support)
	# These bounds come from the "Learning Finite Markov Chains via Adaptive Allocation" article form Sadegh Talebi
	def computeBounds(self, p_estimate, Pk):
		c = 1.1
		bounds = np.zeros((self.nS, self.nA, self.nS))
		self.support = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				n = self.Nk[s, a]
				for ss in range(self.nS):
					if Pk[s, a, ss] == 0:
						bounds[s, a, ss] = 0
					else:
						self.support[s, a] += 1
						tau = np.log((1 / delta) * math.ceil((np.log(n * np.log(c * n)) / np.log(c))))
						b2 = 8 * tau / 3
						p = p_estimate[s, a, ss]
						temp = p * (1 - p) + 2 * np.sqrt((b2 * p * (1 - p)) / n) + (7 * b2) / n
						bound1 = np.sqrt((2 * c * temp * tau) / n) + (4 * tau / 3) / n
						d = self.delta / (2 * self.nS * self.nA)
						bound2 = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
						bounds[s, a, ss] = min(bound1, bound2)
		return bounds
	
	# return an S x A matrice with a 0 where we aggregate (sigma known with high probability) and 1 otherwise
	def computeSetAgg(self):
		test = np.zeros((self.nS, self.nA))
		for s in range(self.nS):
			for a in range(self.nA):
				if (self.support[s, a] < self.sizeSupport[s, a]) or (self.SigmaTest[s, a] == 1):
					test[s, a] = 1
		return test

	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		# First estimate the profile mapping
		self.computeN()
		Pk = self.computeP()
		Rk = self.computeR()
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		Rk_c_true = np.zeros(self.nC_true)
		self.Nk_c_true = np.zeros(self.nC_true)
		for c in range(self.nC_true):
			Rk_c_true[c] = sum([Rk[s, a] for (s, a) in self.C_li[c]])
			self.Nk_c_true[c] = sum([self.Nk[s, a] for (s, a) in self.C_li[c]])
		for s in range(self.nS):
			for a in range(self.nA):
				r_estimate[s, a] = Rk_c_true[self.C_true[s, a]] / max([1, self.Nk_c_true[self.C_true[s, a]]])
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = Pk[s, a, next_s] / div
		self.update_profile_mapping(p_estimate)
		#########
		bounds = self.computeBounds(p_estimate, Pk)
		self.computeSigmaTest(p_estimate, bounds) # used in computeAgg
		test_Agg = self.computeSetAgg()
		#########
		# The additional part of this class: here we exclude from classes the unsampled (s, a)
		self.C = np.zeros((self.nS, self.nA), dtype = int)
		self.nC = 0
		# Aggregation only when test_Agg is equal to zero
		for c in range(self.nC_true):
			agg = -1
			for (s, a) in self.C_li[c]:
				if test_Agg[s, a] == 1:
					self.C[s, a] = self.nC
					self.nC += 1
				else:
					if agg >= 0:
						self.C[s, a] = agg
					else:
						self.C[s, a] = self.nC
						agg = self.nC
						self.nC += 1
		self.vk = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
		# Then initiate the episode as in UCRL2 but using equivalence classes to improve the estimates
		self.computeN_c()
		Pk_c = self.computeP_c()
		p_estimate_c = np.zeros((self.nC, self.nS))
		r_estimate_c = np.zeros(self.nC)
		for s in range(self.nS):
			for a in range(self.nA):# Not the good way to do this..
				r_estimate_c[self.C[s, a]] = r_estimate[s, a]
		for c in range(self.nC):
			div = max([1, self.Nk_c[c]])
			for next_s in range(self.nS):
				p_estimate_c[c, next_s] = Pk_c[c, next_s] / div
		self.distances()
		self.EVI(r_estimate_c, p_estimate_c)
	


class C_UCRL_C14(UCRL2_L_local2):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nC = nC
		self.C = C
		self.agg_set = [[[] for _ in range(nS)] for _ in range(self.nC)]
		self.profile_mapping = np.zeros((nS, nA, nS), dtype = int)
		self.SigmaTest = np.zeros((nS, nA))
		self.C_li = [[] for _ in range(self.nC)]
		self.Rk_c = np.zeros(self.nC)
		self.Nk_c = np.zeros(self.nC)
		for s in range(nS):
			for a in range(nA):
				self.C_li[C[s, a]].append((s, a))
		super().__init__(nS, nA, delta)
	
	def name(self):
		return "C_UCRL_C14"
	
	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				nc = max(1, self.Nk_c[self.C[s, a]])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / nc) * np.log(2 * np.sqrt(nc + 1) / d)) / (2 * nc))
				n = max(1, self.Nk[s, a])
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
	
	
	# for each pair (s, a) comput if on the experimental support there exist two states that could be not well ordered in experimental profile mapping
	# looking at the confidence bounds arounds the transitions probability
	def computeSigmaTest(self, p_estimate):
		self.SigmaTest = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for i_ss in range(self.nS - 1):
					ss = self.profile_mapping[s, a, i_ss]
					for i_p_ss in range(ss + 1, self.nS):
						p_ss = self.profile_mapping[s, a, i_p_ss]
						test1 = (p_estimate[s, a, ss] + self.p_distances[s, a, ss] >= p_estimate[s, a, p_ss] - self.p_distances[s, a, p_ss])
						test2 = (p_estimate[s, a, ss] + self.p_distances[s, a, ss] <= p_estimate[s, a, p_ss] + self.p_distances[s, a, p_ss])
						test3 = (p_estimate[s, a, ss] - self.p_distances[s, a, ss] >= p_estimate[s, a, p_ss] - self.p_distances[s, a, p_ss])
						test4 = (p_estimate[s, a, ss] - self.p_distances[s, a, ss] <= p_estimate[s, a, p_ss] + self.p_distances[s, a, p_ss])
						case1 = test1 and test2
						case2 = test3 and test4
						case3 = not (test2 and test3)
						if (case1 or case2 or case3):
							self.SigmaTest[s, a,  i_ss] = self.SigmaTest[s, a, i_p_ss] = 1
	
	# Give sets of transition probability that can be aggregated (sigma known with high probability which implies that there is no bias
	# to take into account in the confidence intervals)	
	def aggregate(self):# To be updated and ADD THE UPDATE OF C-li
		self.agg_set = [[[] for _ in range(self.nS)] for _ in range(self.nC)]
		test = 1
		for c in range(self.nC):
			if len(self.C_li[c]) > 1:
				for (s, a) in self.C_li[c]:
					while 0 in self.SigmaTest[s, a]:
						test = 0
						i_ss = list(self.SigmaTest[s, a]).index(0)
						self.SigmaTest[s, a, i_ss] = 1
						self.agg_set[c][i_ss].append((s, a))
		if test == 0:
			print("Aggregate at time t = ", self.t)
					
	
	# Having the result of the self.aggregate function update the concerned transitions
	def aggregate_p_estimates(self):
		for c in range(self.nC):
			for i_ss in range(self.nS):
				li = self.agg_set[c][i_ss]
				if len(li) > 1:
					p_ss = sum([self.Nk[s, a] * self.p_estimate[s, a, self.profile_mapping[s, a, i_ss]] for (s, a) in li]) / sum([self.Nk[s, a]
																															for (s, a) in li])
					for (s, a) in li:
						self.p_estimate[s, a, self.profile_mapping[s, a, i_ss]] = p_ss
		return self.p_estimate
	
	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])

	# Having the result of the self.aggregate function update the concerned confidence bounds on the transitions (denoted as p_distances)
	def aggregate_p_distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for c in range(self.nC):
			for i_ss in range(self.nS):
				li = self.agg_set[c][i_ss]
				if len(li) > 1:
					n =  sum([self.Nk[s, a]	for (s, a) in li])
					d_ss = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					for (s, a) in li:
						self.p_distances[s, a, self.profile_mapping[s, a, i_ss]] = d_ss
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		# First estimate the profile mapping
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.Rk_c = np.zeros(self.nC)
		self.Nk_c = np.zeros(self.nC)
		for c in range(self.nC):
			self.Rk_c[c] = sum([self.Rk[s, a] for (s, a) in self.C_li[c]])
			self.Nk_c[c] = sum([self.Nk[s, a] for (s, a) in self.C_li[c]])
		for s in range(self.nS):
			for a in range(self.nA):
				r_estimate[s, a] = self.Rk_c[self.C[s, a]] / max([1, self.Nk_c[self.C[s, a]]])
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances()
		#########
		self.update_profile_mapping(p_estimate)
		self.computeSigmaTest(p_estimate)
		# Aggregation only when sigma(next_s) is known with high proabability
		self.aggregate()
		p_estimate = self.aggregate_p_estimates()
		self.aggregate_p_distances()
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)
		
		
class C_UCRL_C14_Nrwd(C_UCRL_C14):
	def name(self):
		return "C_UCRL_C14_no_reward"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				nc = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / nc) * np.log(2 * np.sqrt(nc + 1) / d)) / (2 * nc))
				n = max(1, self.Nk[s, a])
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))



# Combination of leanrners.UCRL2_Lplus_UCRL2_Lplus_local3 and te previous C_UCRL_C14_Nrwd (only to check if what is done on the transition is
# working, otherwise aggregating the reward is obviously a good thing)
class C_UCRL_C14_Nrwd_Lplus_local3(UCRL2_Lplus_local3):
	def __init__(self,nS, nA, delta, C, nC):
		print("Initialize C_UCRL_C with : C = ", C, " and nC = ", nC)
		self.nC = nC
		self.C = C
		self.agg_set = [[[] for _ in range(nS)] for _ in range(self.nC)]
		self.profile_mapping = np.zeros((nS, nA, nS), dtype = int)
		self.SigmaTest = np.zeros((nS, nA))
		self.C_li = [[] for _ in range(self.nC)]
		self.Rk_c = np.zeros(self.nC)
		self.Nk_c = np.zeros(self.nC)
		for s in range(nS):
			for a in range(nA):
				self.C_li[C[s, a]].append((s, a))
		super().__init__(nS, nA, delta)
	
	def name(self):
		return "C_UCRL_C14_notReward_Lplus_local3"
	
	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])
	
	
	# for each pair (s, a) comput if on the experimental support there exist two states that could be not well ordered in experimental profile mapping
	# looking at the confidence bounds arounds the transitions probability
	def computeSigmaTest(self, p_estimate):
		self.SigmaTest = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for i_ss in range(self.nS):# - 1):
					ss = self.profile_mapping[s, a, i_ss]
					for i_p_ss in range(self.nS):#(ss + 1, self.nS):
						p_ss = self.profile_mapping[s, a, i_p_ss]
						test1 = (p_estimate[s, a, ss] + self.p_distances[s, a, ss, 1] >= p_estimate[s, a, p_ss] - self.p_distances[s, a, p_ss, 0])
						#test2 = (p_estimate[s, a, ss] + self.p_distances[s, a, ss, 1] <= p_estimate[s, a, p_ss] + self.p_distances[s, a, p_ss, 1])
						#test3 = (p_estimate[s, a, ss] - self.p_distances[s, a, ss, 0] >= p_estimate[s, a, p_ss] - self.p_distances[s, a, p_ss, 0])
						test4 = (p_estimate[s, a, ss] - self.p_distances[s, a, ss, 0] <= p_estimate[s, a, p_ss] + self.p_distances[s, a, p_ss, 1])
						#case1 = test1 and test2
						#case2 = test3 and test4
						#case3 = not (test2 and test3)
						if (test1 and test4) and (p_ss != ss):#(case1 or case2 or case3):
							self.SigmaTest[s, a,  i_ss] = 1
							break
							#self.SigmaTest[s, a, i_p_ss] = 1
	
	# Give sets of transition probability that can be aggregated (sigma known with high probability which implies that there is no bias
	# to take into account in the confidence intervals)	
	def aggregate(self):# To be updated and ADD THE UPDATE OF C-li
		self.agg_set = [[[] for _ in range(self.nS)] for _ in range(self.nC)]
		test = 1
		for c in range(self.nC):
			if len(self.C_li[c]) > 1:
				for (s, a) in self.C_li[c]:
					while 0 in self.SigmaTest[s, a]:
						test = 0
						i_ss = list(self.SigmaTest[s, a]).index(0)
						self.SigmaTest[s, a, i_ss] = 1
						self.agg_set[c][i_ss].append((s, a))
		#if test == 0:
		#	print("Aggregate at time t = ", self.t)
					
	
	# Having the result of the self.aggregate function update the concerned transitions
	def aggregate_p_estimates(self):
		for c in range(self.nC):
			for i_ss in range(self.nS):
				li = self.agg_set[c][i_ss]
				if len(li) > 1:
					p_ss = sum([self.Nk[s, a] * self.p_estimate[s, a, self.profile_mapping[s, a, i_ss]] for (s, a) in li]) / sum([self.Nk[s, a]
																															for (s, a) in li])
					for (s, a) in li:
						self.p_estimate[s, a, self.profile_mapping[s, a, i_ss]] = p_ss
		return self.p_estimate
	
	# To update the current profile mapping (as defined in the paper).
	def update_profile_mapping(self, p):
		for s in range(self.nS):
			for a in range(self.nA):
				self.profile_mapping[s, a] = np.argsort(p[s, a])

	# Having the result of the self.aggregate function update the concerned confidence bounds on the transitions (denoted as p_distances)
	def aggregate_p_distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for c in range(self.nC):
			for i_ss in range(self.nS):
				li = self.agg_set[c][i_ss]
				#print(len(li))
				if len(li) > 1:
					n =  sum([self.Nk[s, a]	for (s, a) in li])
					(s0, a0) = li[0]
					d_ss_up = self.bound_upper(self.p_estimate[s0, a0, self.profile_mapping[s0, a0, i_ss]], n, d)
					d_ss_lo = self.bound_lower(self.p_estimate[s0, a0, self.profile_mapping[s0, a0, i_ss]], n, d)
					for (s, a) in li:
						self.p_distances[s, a, self.profile_mapping[s, a, i_ss], 0] = d_ss_lo
						self.p_distances[s, a, self.profile_mapping[s, a, i_ss], 1] = d_ss_up
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		# First estimate the profile mapping
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.Rk_c = np.zeros(self.nC)
		self.Nk_c = np.zeros(self.nC)
		for c in range(self.nC):
			self.Rk_c[c] = sum([self.Rk[s, a] for (s, a) in self.C_li[c]])
			self.Nk_c[c] = sum([self.Nk[s, a] for (s, a) in self.C_li[c]])
		for s in range(self.nS):
			for a in range(self.nA):
				r_estimate[s, a] = self.Rk_c[self.C[s, a]] / max([1, self.Nk_c[self.C[s, a]]])
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances(p_estimate, r_estimate)
		#########
		self.update_profile_mapping(p_estimate)
		self.computeSigmaTest(p_estimate)
		# Aggregation only when sigma(next_s) is known with high proabability
		self.aggregate()
		p_estimate = self.aggregate_p_estimates()
		self.aggregate_p_distances()# Use the aggregated p_estimate
		self.supports = self.computeSupports(p_estimate)
		#print(self.t)
		#print(self.Nk)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)


class C_UCRL_C14_Lplus_local3(C_UCRL_C14_Nrwd_Lplus_local3):
	def name(self):
		return "C_UCRL_C14_Lplus_local3"
	
	def distances(self, p_estimate, r_estimate):
		d = self.delta / (2 * self.nS * self.nA)
		dc = self.delta / (2 * self.nC)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				nc = max(1, self.Nk_c[self.C[s, a]])
				self.r_distances[s, a] = self.bound_upper(r_estimate[s, a], nc, dc)
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s, 0] = self.bound_lower(p_estimate[s, a, next_s], n, d)
					self.p_distances[s, a, next_s, 1] = self.bound_upper(p_estimate[s, a, next_s], n, d)




















































































# IMPLEMENTATIO NOT FINISHED -> But not usefull because this algorithm cannot learn..
# Second version with only classes known but using the most optimistic profile in all possible permutation (computationally heavy, don't run it on big
# MDP) because of the unrealistic computational cost of the idea, the class is not finished (and I think that this idea should not even work, if at
# each episode we take the most optimistic permutation among alll possible permutation we will just wrongly consider that all states could have their
# highest transition probability to go to the state with highest value (and do this mistake for ever)...)
class C_UCRL_C_allpermut(C_UCRL_C):
	def __init__(self,nS, nA, delta, C, nC):
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
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.C_li = [[] for _ in range(self.nC)]
		for s in range(self.nS):
			for a in range(self.nA):
				self.C_li[C[s, a]].append((s, a))
	
	# Alternative version returning the maximum considering transitions AND porfile mapping
	def max_proba(self, p_estimate, sorted_indices, c, s, a):
		min1 = min([1, p_estimate[c, int(list(self.profile_mapping[s, a]).index(sorted_indices[-1]))] + (self.p_distances[c] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			#max_p = [p_estimate[c, int(i)] for i in self.profile_mapping[s, a]]
			for ss in range(self.nS):
				max_p[ss] = p_estimate[c, list(self.profile_mapping[s, a]).index(ss)]
			max_p[sorted_indices[-1]] += self.p_distances[c] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate_c, p_estimate, epsilon = 0.01, max_iter = 100):
		current_iter = 0
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		for sigma in permut:
			while True:
				current_iter += 1
				for s in range(self.nS):
					for a in range(self.nA):
						c = self.C[s, a]
						sigma, max_p = self.max_proba(p_estimate_c, sorted_indices, c, s, a)
						temp = min((1, r_estimate_c[c] + self.r_distances[c])) + sum([u * p for (u, p) in zip(u0, max_p)])
						if (a == 0) or (temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
				diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
				if current_iter > max_iter:
					print("No convergence in the EVI")
					break
				if (max(diff) - min(diff)) < epsilon:
					break
				else:
					u0 = u1
					u1 = np.zeros(self.nS)
					sorted_indices = np.argsort(u0)
	
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
		self.computeN_c()
		Rk_c = self.computeR_c()
		r_estimate_c = np.zeros(self.nC)
		for c in range(self.nC):
			div = max([1, self.Nk_c[c]])
			r_estimate_c[c] = Rk_c[c] / div
		self.distances()
		self.EVI(r_estimate_c, p_estimate)