from learners.UCRL import *
import scipy as sp
import numpy as np

# KL-UCRL is an improvement of UCRL2 introduced by Filippi et al. 2011
# This class proposes an implementation of this algorithm, it seems usefull to know that the algorithm proposed in the paper cannot be implemented as
# proposed. Some modifications have to be done (and are done here) in order to prevent some problems as: division by 0 or log(0) in function f and
# newton optimization on constant function.
class KL_UCRL(UCRL2):
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
		self.p_distances = np.zeros((self.nS, self.nA))
	
	def name(self):
		return "KL-UCRL"
	
	# Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
	# KL-UCRL variant (Cp and Cr are difined as constrained constants in the paper of Filippi et al 2011, here we use the one used on the proofs
	# provided by the paper (with 2 instead of T at the initialization to prevent div by 0).
	def distances(self):
		B = np.log((2 * np.exp(1) * (self.nS)**2 * self.nA * np.log(max([2, self.t]))) / self.delta)
		Cp = self.nS * (B + np.log(B + 1 / np.log(max([2, self.t]))) * (1 + 1 / (B + 1 / np.log(max([2, self.t])))))
		Cr = np.sqrt((np.log(4 * self.nS * self.nA * np.log(max([2, self.t])) / self.delta)) / 1.99)
		for s in range(self.nS):
			for a in range(self.nA):
				self.r_distances[s, a] = Cr / np.sqrt(max([1, self.Nk[s, a]]))
				self.p_distances[s, a] = Cp / (max([1, self.Nk[s, a]]))

	# Key function of the problem -> solving the maximization problem is essentially based on finding roots of this function.
	def f(self, nu, p, V, Z_): # notations of the paper
		sum1 = 0
		sum2 = 0
		for i in Z_:
			if nu == V[i]:
				return - 10**10
			sum1 += p[i] * np.log(nu - V[i])
			sum2 += p[i] / (nu - V[i])
		if sum2 <= 0:
			return - 10**10
		return sum1 + np.log(sum2)
	
	# Derivative of f, used in newton optimization.
	def diff_f(self, nu, p, V, Z_, epsilon = 0):
		sum1 = 0
		sum2 = 0
		for i in range(len(p)):
			if i in Z_:
				sum1 += p[i] / (nu - V[i])
				sum2 += p[i] / (nu - V[i])**2
		return sum1 - sum2 / sum1

	# The maximization algorithm proposed by Filippi et al. 2011.
	# Inspired (for error preventing) from a Matlab Code provided by Mohammad Sadegh Talebi.
	# Exotics inputs:
	#	tau our approximation of 0
	#	max_iter maximmum number of iterations on newton optimization
	#	tol precision required in newton optimization
	def MaxKL(self, p_estimate, u0, s, a, tau = 10**(-8), max_iter = 10, tol = 10**(-5)):
		degenerate = False # used to catch some errors
		Z, Z_, argmax = [], [], []
		maxV = max(u0)
		q = np.zeros(self.nS)
		for i in range(self.nS):
			if u0[i] == maxV:
				argmax.append(i)
			if p_estimate[s, a, i] > tau:
				Z_.append(i)
			else:
				Z.append(i)
		I = []
		test0 = False
		for i in argmax:
			if i in Z:
				I.append(i)
				test0 = True
		if test0:
			test = [(self.f(u0[i], p_estimate[s, a], u0, Z_) < self.p_distances[s, a]) for i in I]
		else:
			test = [False]
		if (True in test) and (maxV > 0): # I must not and cannot be empty if this is true.
			for i in range(len(test)):
				if test[i]: # it has to happen because of previous if
					nu = u0[I[i]]
					break
			r = 1 - np.exp(self.f(nu, p_estimate[s, a], u0, Z_) - self.p_distances[s, a])
			for i in I: # We want sum(q[i]) for i in I = r.
				q[i] = r / len(I)
		else:
			if len(Z) >= self.nS - 1: # To prevent the algorithm from running the Newton optimization on a constant or undefined function.
				degenerate = True
				q = p_estimate[s, a]
			else:
				VZ_ = []
				for i in range(len(u0)):
					if p_estimate[s, a, i] > tau:
						VZ_.append(u0[i])
				nu0 = 1.1 * max(VZ_)  # This choice of initialization is based on the Matlab Code provided by Mohammad Sadegh Talebi, the one
				# provided by the paper leads to many errors while T is small.
				# about the following (unused) definition of nu0 see apendix B of Filippi et al 2011
				#nu0 = np.sqrt((sum([p_estimate[s, a, i] * u0[i]**2 for i in range(self.nS)]) -
				#			  (sum([p_estimate[s, a, i] * u0[i] for i in range(self.nS)]))**2) / (2 * self.p_distances[s, a]))
				r = 0
				nu1 = 0
				err_nu = 10**10
				k = 1
				while (err_nu >= tol) and (k < max_iter):
					nu1 = nu0 - (self.f(nu0, p_estimate[s, a], u0, Z_) - self.p_distances[s, a]) / (self.diff_f(nu0, p_estimate[s, a], u0, Z_))
					if nu1 < max(VZ_):# f defined on ]max(VZ_); +inf[ we have to prevent newton optimization from going out from the definition interval
						nu1 = max(VZ_) + tol
						nu0 = nu1
						k += 1
						break
					else:
						err_nu = np.abs(nu1 - nu0)
						k += 1
						nu0 = nu1
				nu = nu0
		if not degenerate:
			q_tilde = np.zeros(self.nS)
			for i in Z_:
				if nu == u0[i]:
					q_tilde[i] = p_estimate[s, a, i] * 10**10
				else:
					q_tilde[i] = p_estimate[s, a, i] / (nu - u0[i])
			sum_q_tilde = sum(q_tilde)
			for i in Z_:
				q[i] = ((1 - r) * q_tilde[i]) / sum_q_tilde
		return q

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# In KL-UCRL MaxKL used instead of max_proba (see Filippi et al. 2011).
	def EVI(self, r_estimate, p_estimate, epsilon = 0.1):
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		tau = 10**(-6)
		maxiter = 1000
		niter = 0
		while True:
			for s in range(self.nS):
				test0 = (False in [tau > u for u in u0]) # Test u0 != [0,..., 0]
				for a in range(self.nA):
					if not test0: # MaxKL cannot run with V = [0, 0,..., 0, 0] because function f undifined in this case.
						max_p = p_estimate[s, a]
					else:
						max_p = self.MaxKL(p_estimate, u0, s, a)
					temp = r_estimate[s, a] + self.r_distances[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [x - y for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
			if niter > maxiter:
				break
			else:
				niter += 1
				





class KL_UCRL_L(KL_UCRL):
	def name(self):
		return "KL-UCRL-L"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
