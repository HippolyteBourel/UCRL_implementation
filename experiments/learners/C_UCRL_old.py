from learners.C_UCRL_C import *
import scipy as sp
import numpy as np
import copy as cp

# Work in progress.

# C_UCRL is the algorithm introduced by Maillard and Asadi 2018, it is the more realistic version were C (the classes) and sigma (the profile
# mapping) are unknown. It extends C_UCRL_C which is the implementation of the algorihtm C_UCRL(C) of the paper, using clustering in order to
# estimate C.
class C_UCRL_old(C_UCRL_C):
	def __init__(self,nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.nC = nS * nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros(self.nC)
		self.Nk = np.zeros(self.nC)
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros(self.nC)
		self.p_distances = np.zeros(self.nC)
		self.C = np.zeros(1)
		self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
		self.deleted = [] # Used during the clustering algorithm to memorize deleted classes.

	def name(self):
		return "C_UCRL_old"
	
	# Norme 1 of the difference between 2 vectors of same size.
	def diffNorme1(self, v1, v2):
		res = 0
		for i in range(len(v1)):
			res += abs(v1[i] - v2[i])
		return res
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))
	
	# bW function from the paper, it's the Laplace confidence bound used for the transition probability.
	# K is the support of p(.|s, a), but as the paper by hypothesis we have no prior information on the support so nS is used by default.
	def bW(self, n, delta, K = -1):
		if K == -1:
			K = self.nS
		temp = 2 * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) * (2**K - 2) / delta)
		return np.sqrt(temp / n)

	# Auxiliary function for the clustering algorithm, used to find the closest cluster.
	def near(self, index, p_estimate, r_estimate, p_estimate_init, r_estimate_init, Nk_init):
		k = -1
		dist_min = float("inf")
		current_nC = self.nC - len(self.deleted)
		delta = self.delta / current_nC
		for i in range(self.nC):
			if (i != index) and (self.Nk_c[i] != 0):
				dist_p = self.diffNorme1(p_estimate[i], p_estimate[index]) - self.bW(self.Nk_c[i], delta) - self.bW(self.Nk_c[index], delta)
				dist_r = abs(r_estimate[i] - r_estimate[index]) - self.bH(self.Nk_c[i], delta) - self.bH(self.Nk_c[index], delta)
				dist = max([dist_p, dist_r])
				if (dist < 0) and (dist < dist_min) and self.isValid(i, index, p_estimate_init, r_estimate_init, Nk_init):
					dist_min = dist
					k = i
		return k

	
	# Auxiliary function for the clustering algorithm, used to combines two cluster centers. Implies modifications on self.C.
	def merge(self, i, j, size, p_estimate, r_estimate):
		for s in range(self.nS):
			p_estimate[i, s] = (size[i] * p_estimate[i, s] + size[j] * p_estimate[j, s]) / (size[i] + size[j])
		r_estimate[i] = (size[i] * r_estimate[i] + size[j] * r_estimate[j]) / (size[i] + size[j])
		size[i] += size[j]
		self.Nk_c[i] += self.Nk_c[j]
		self.Nk_c[j] = 0
		for s in range(self.nS):
			for a in range(self.nA):
				if self.C[s, a] == j:
					self.C[s, a] = i
		self.deleted.append(j)
		return size, p_estimate, r_estimate
		
	# li is a list of classes (int representing the classes), get_pairs return the list of lists with the state-action pairs in these classes.
	def get_pairs(self, li):
		res = [[] for _ in range(len(li))]
		for s in range(self.nS):
			for a in range(self.nA):
				c = self.C[s, a]
				if c in li:
					res[li.index(c)].append(s * self.nA + a)
		return res

	# Auxiliary function for the clustering algorithm, used to check both cluster points to have a valid distance.
	def isValid(self, i, j, p_estimate_init, r_estimate_init, Nk_init):
		[samples_i, samples_j] = self.get_pairs([i, j])
		delta = self.delta / self.nC # When this function is called we have self.nC = nS * nA
		for sample_i in samples_i:
			for sample_j in samples_j:
				dist_p = (self.diffNorme1(p_estimate_init[sample_i], p_estimate_init[sample_j])
					- self.bW(Nk_init[sample_i], delta) - self.bW(Nk_init[sample_j], delta))
				dist_r = (abs(r_estimate_init[sample_i] - r_estimate_init[sample_j])
						  - self.bH(Nk_init[sample_i], delta) - self.bH(Nk_init[sample_j], delta))
				dist = max([dist_p, dist_r])
				if dist > 0:
					return False
		#######################################################################################
		############################ MODIFICATION OF THE ALGORITHM ############################
		#######################################################################################
		return True
	
	# The clustering algorithm introduced in the paper, used to estimate the classes.
	def clustering(self, p_estimate, r_estimate):
		size = np.array([1 for _ in range(self.nC)])
		p_estimate_init, r_estimate_init = cp.deepcopy(p_estimate), cp.deepcopy(r_estimate)
		Nk_init = cp.deepcopy(self.Nk_c)
		changed = True
		while changed:
			changed = False
			ordering  = (np.argsort(self.Nk_c))[::-1]
			for i in ordering:
				if self.Nk_c[i] == 0:
					break
				k = self.near(i, p_estimate, r_estimate, p_estimate_init, r_estimate_init, Nk_init)
				if k != -1:
					size, p_estimate, r_estimate = self.merge(k, i, size, p_estimate, r_estimate) # cluster that should be deleted are add into the list self.deleted, and their NK = 0
					changed = True
	
	# To start a new episode (init var, computes estmates and run EVI). C_UCRL version, the clustering algorithm is used here.
	# It is probably possible (and quite easy) to speed up this function with some optimizations... -> by using self.deleted for example
	def new_episode(self):
		# First initialization for the clustering and then running it.
		self.nC = self.nS * self.nA
		self.C = np.arange(self.nC).reshape((self.nS, self.nA))
		self.vk = np.zeros(self.nC)
		self.computeN()
		self.computeN_c()
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
		if self.t > 1:
			self.clustering(p_estimate_c, r_estimate_c)
		# Having the clustering we now put it "in good format" (which means redifining C and nC properly). This should be "optimizable" with self.deleted.
		self.nC -= len(self.deleted)
		dico = []
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
