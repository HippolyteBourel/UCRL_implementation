import scipy as sp
import numpy as np
import copy
import math

# UCRL3 is what we defined as a local UCRL (with element-wise confidence bounds), and a near optimistic optimisation in the inner
# maximization of the Extended Value Iteration. (it the same algorithm as \UCR_Lplus\UCRL2_Lplus_local3 without inheritance)
class UCRL3:
	def __init__(self, nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		#self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.span = []
		self.p_distances = np.zeros((nS, nA, nS, 2))
	
	def name(self):
		return "UCRL3"
	
	###### Computation of confidences intervams (named distances in implementation) ######
	# Bounds are estimated using 16 steps of dichotomic search

	# Auxiliary function to compute upper confidence bounds
	def beta_minus(self, p, n, d):
		g = (1 / 2 - p) / np.log(1 / p - 1)
		return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

	# Auxiliary function to compute lower confidence bounds
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

	# Auxiliary for bernstein's bounds
	def beta(self, n, delta):
		temp = 2 * np.log(np.log(max((np.exp(1), n)))) + np.log(3 / delta)
		return temp

	# Tigther than Laplace bounds confidence interval, depending on the estimate value, we have for each transition of each pair 2 bounds
	# The lower (index 0) one and the upper (index 1) one.
	# Laplace-g(p) bounds, original distances function of this algo (bring the global experimental result)
	#def distances(self, p_estimate, r_estimate):
	#	d = self.delta / (2 * self.nS * self.nA)
	#	for s in range(self.nS):
	#		for a in range(self.nA):
	#			n = max(1, self.Nk[s, a])
	#			self.r_distances[s, a] = self.bound_upper(r_estimate[s, a], n, d)
	#			for next_s in range(self.nS):
	#				self.p_distances[s, a, next_s, 0] = self.bound_lower(p_estimate[s, a, next_s], n, d)
	#				self.p_distances[s, a, next_s, 1] = self.bound_upper(p_estimate[s, a, next_s], n, d)

	# Bernstein bounds code (cannot be use as it is, upper and lower undistingushed currently)
	#def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		delta = d
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				r_bernstein = np.sqrt(self.r_estimate[s, a] * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
				self.r_distances[s, a] = r_bernstein
				for next_s in range(self.nS):
					p = self.p_estimate[s, a, next_s]
					p_bernstein = np.sqrt(p * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
					self.p_distances[s, a, next_s, 0] = p_bernstein
					self.p_distances[s, a, next_s, 1] = p_bernstein

	# Minimum between previous Laplace-g(p) and Bernstein bounds (best bounds so far, but slow down the computation a bit)
	def distances(self, p_estimate, r_estimate):
		d = self.delta / (2 * self.nS * self.nA)
		delta = d
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				r_bernstein = np.sqrt(self.r_estimate[s, a] * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
				self.r_distances[s, a] = min((r_bernstein, self.bound_upper(r_estimate[s, a], n, d)))
				for next_s in range(self.nS):
					p = self.p_estimate[s, a, next_s]
					p_bernstein = np.sqrt(p * self.beta(n, delta) / n) + self.beta(n, delta) / (3 * n)
					self.p_distances[s, a, next_s, 0] = min((p_bernstein, self.bound_lower(p, n, d)))
					self.p_distances[s, a, next_s, 1] = min((p_bernstein, self.bound_upper(p, n, d)))

	###### Functions used to initialize an episode ######
	
	# Auxilliary function used to compute the Fplus (inside the extended support)
	# (compute the inner maximization in order to compute Fplus)
	def aux_compute_Fplus_in(self, s, a, u, p_estimate, sorted_indices, p, idx):
		sum_p = sum(p)
		while sum_p > 1 and idx < self.nS:
			temp = max((0, p[sorted_indices[idx]] - sum_p + 1, p_estimate[s, a, sorted_indices[idx]]
						- self.p_distances[s, a, sorted_indices[idx], 0]))
			sum_p -= p[sorted_indices[idx]] - temp
			p[sorted_indices[idx]] = temp
			if p[sorted_indices[idx]] - temp <= 0:
				idx += 1
		return p, idx
	
	# Auxilliary function used to compute the Fplus (outside of the extended support)
	# (compute the inner maximization in order to compute Fplus)
	def aux_compute_Fplus_out(self, s, a, u, p_estimate, sorted_indices, p, idx, support):
		sum_p = sum(p)
		while sum_p < 1 and idx >= 0:
			if not support[sorted_indices[idx]]:
				temp = min((self.p_distances[s, a, sorted_indices[idx], 1], max(0, 1 - sum_p)))
				sum_p += temp
				p[sorted_indices[idx]] = temp
				if self.p_distances[s, a, sorted_indices[idx], 1] >= 1 - sum_p:
					idx -= 1
			else:
				idx -= 1
		return p, idx
	

	
	# Auxiliary function for the inner maximization of the EVI, dealing with the Near Optimistic Optimization based on the support
	# for transition function used in the inner maximization
	def computeSupport(self, s, a, u, p_estimate, sorted_indices):
		support = copy.deepcopy(self.supports[s, a])
		emp_support = sum(support)
		min_u = min(u)
		span = max(u) - min_u
		support[sorted_indices[self.nS - 1]] = True
		l = 1
		
		#### With sum of beta plus ####
		#temp = sum([(u[next_s] - min_u) * (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1]) * (1 - support[next_s])
		#			for next_s in range(self.nS)])
		#temp2 = sum([(u[next_s] - min_u) * (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1]) * support[next_s]
		#			for next_s in range(self.nS)])
		
		#### With epsilon (need to uncomment temp and replace temp2 by epsilon in following while) ####
		#c = 10
		#N = max(1, max([max(self.Nk[s]) for s in range(self.nS)]))
		#epsilon = c * max(1, span) * sum(support) / (N**(2/3))
		#(self.nS * max(1, span) / np.log(self.t))
		
		#### Using Fplus ####
		p_in = [min((1, (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1]) * support[next_s])) for next_s in range(self.nS)]
		p_out = [0 for _ in range(self.nS)]
		idx_in = 0
		idx_out = self.nS - 1
		p_in, idx_in = self.aux_compute_Fplus_in(s, a, u, p_estimate, sorted_indices, p_in, idx_in)
		p_out, idx_out = self.aux_compute_Fplus_out(s, a, u, p_estimate, sorted_indices, p_out, idx_out, support)
		temp = sum([p_in[next_s] * (u[next_s] - min_u) for next_s in range(self.nS)])
		temp2 = sum([p_out[next_s] * (u[next_s ]- min_u) for next_s in range(self.nS)])
		
		while temp2 > temp and l < self.nS - 1:
		#while temp2 > min(epsilon, temp) and l < self.nS - 1:
			while support[sorted_indices[self.nS - 1 - l]] and l < self.nS - 1: # To add something that is actually not in the empirical support
				l += 1
			next_s = sorted_indices[self.nS - 1 - l]
			support[next_s] = True
			
			##### With sum of beta plus ####
			#temp -= (u[next_s] - min_u) * (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1])
			#temp2 += (u[next_s] - min_u) * (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1])
			
			##### With epsilon ####
			#epsilon = c * max(1, span) * sum(support) / (N**(2/3))
			
			##### With Fplus #####
			if self.nS - 1 - l > idx_in:
				p_in[next_s] = (p_estimate[s, a, next_s] + self.p_distances[s, a, next_s, 1])
				p_in, idx_in = self.aux_compute_Fplus_in(s, a, u, p_estimate, sorted_indices, p_in, idx_in)
			p_out[next_s] = 0
			p_out, idx_out = self.aux_compute_Fplus_out(s, a, u, p_estimate, sorted_indices, p_out, idx_out, support)
			temp = sum([p_in[i] * (u[i] - min_u) for i in range(self.nS)])
			temp2 = sum([p_out[i] * (u[i] - min_u) for i in range(self.nS)])
			
		#print("At time ", self.t, " l = ", l)
		return support
	
	# Inner maximization of the Extended Value Iteration
	def max_proba(self, p_estimate, sorted_indices, s, a, support, epsilon = 10**(-8)):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > epsilon) and (l <= self.nS - 1):
			idx = self.nS - 1 - l
			idx = sorted_indices[idx]
			if (l == 0) or support[idx]:
				new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
				max_p[idx] += new_delta
				delta += - new_delta
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
					support = self.computeSupport(s, a, u0, p_estimate, sorted_indices)
					max_p = self.max_proba(p_estimate, sorted_indices, s, a, support)
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

	def computeEmpiricalSupports(self, p_estimate):
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
		self.supports = self.computeEmpiricalSupports(p_estimate)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)
	
	
	###### Steps and updates functions ######
	
	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]
	
	# Auxiliary function to update R the accumulated reward.
	def updateR(self):
		self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]
	
	# Auxiliary function to update P the transitions count.
	def updateP(self):
		self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1
	
	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.new_episode()
		self.u = np.zeros(self.nS)
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.span = [0]

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.updateR()
		self.t += 1

# UCRL3 with modified stopping criterion based on hypothesis testing
class UCRL3_MSC(UCRL3):
	def name(self):
		return "UCRL3-MSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			#Pk = self.Pk[s, a, state]
			n = max([1, self.Nk[s, a] + self.vk[s, a]])
			#d = self.delta / (2 * self.nS * self.nA)
			temp1 = self.p_estimate[s, a, state] - self.Pk[s, a, state] / n
			temp0 = - self.p_distances[s, a, state, 0]#(1 + self.epsilon) * self.bH(n, d)
			temp2 = self.p_distances[s, a, state, 1]
			if (temp1  > temp2) or (temp0 > temp1) or ( self.vk[state, action] >= max([1, self.Nk[state, action]])): # Stoppping criterion
				self.new_episode()
		action  = self.policy[state]
		return action
	

# UCRL3 with nested loops in the EVI
class UCRL3_nested(UCRL3):
	def name(self):
		return "UCRL3-nested"
	
	# EVI with nested loops
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000, nup_steps = 10):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		itera = 0
		max_p = np.zeros((self.nS, self.nA, self.nS))
		while True:
			sorted_indices = np.argsort(u0)
			for s in range(self.nS):
				for a in range(self.nA):
					support = self.computeSupport(s, a, u0, p_estimate, sorted_indices)
					max_p[s, a] = self.max_proba(p_estimate, sorted_indices, s, a, support)
			for _ in range(nup_steps):
				for s in range(self.nS):
					for a in range(self.nA):
						temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p[s, a])])
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
				diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
				if (max(diff) - min(diff)) < epsilon:
					self.u = u1
					self.span.append(max(u1) - min(u1))
					return None #replace the usual break, end the EVI
				elif itera > max_iter:
					print("No convergence in the EVI at time : ", self.t)
					return None #replace the usual break, end the EVI
				else:
					u0 = u1
					u1 = np.zeros(self.nS)
					itera += 1