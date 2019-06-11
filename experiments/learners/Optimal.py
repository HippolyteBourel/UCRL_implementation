import numpy as np
import copy as cp

def build_opti(name, env, nS, nA):
	if name == "RiverSwim":
		return Opti_swimmer(env)
	else:
		return Opti_learner(env, nS, nA)

class Opti_swimmer:
	def __init__(self,env):
		self.env=env

	def name(self):
		return "Opti_swimmer"

	def reset(self,inistate):
		()

	def play(self,state):
		return 0

	def update(self, state, action, reward, observation):
		()

# The following class has as inputs env the MDP, nS its number of states and nA its number of actions.
# This class run a policy iteration in order to estimate the optimal policy on the given MDP (could be usefull to compute the regret for example).
# The two last inputs epsilon and max_iter are respectively the precision asked in the value iteration and the maximum number of iterations
# in the policy iteration.
class Opti_learner:
	def __init__(self, env, nS, nA, epsilon = 0.001, max_iter = 100):
		self.env = env
		self.nS = nS
		self.nA = nA
		self.epsilon = epsilon
		self.not_converged = True
		self.transitions = np.zeros((self.nS, self.nA, self.nS))
		self.rewards = np.zeros((self.nS, self.nA)) # Precily these are the mean reward for given state and action, see DiscreteMDP.py
		for s in range(self.nS):
			for a in range(self.nA):
				self.transitions[s, a] = self.env.getTransition(s, a)
				self.rewards[s, a] = self.env.getReward(s, a)
		self.policy = np.zeros(nS, dtype=int)
		self.learn(max_iter = max_iter) # learn the optimal policy by policy iteration so could be long depending on the MDP
		
	def name(self):
		return "Opti_leanrner"
	
	def reset(self, inistate):
		()
		
	def play(self, state):
		return self.policy[state]
	
	def update(self, state, action, reward, observation):
		()
	
	def update_policy(self, V):
		old = cp.deepcopy(self.policy)
		for s in range(self.nS):
			li = np.zeros(self.nA)
			for a in range(self.nA):
				li[a] = sum([self.transitions[s, a, next_s] * V[next_s] for next_s in range(self.nS)])
			self.policy[s] = int(np.argmax(li))
		for s in range(self.nS):
			self.not_converged = self.not_converged and (self.policy[s] != old[s])
	
	def value_it(self, max_iter = 1000, first_time = False):
		V1 = np.zeros(self.nS)
		delta = 100
		itera = 0
		while (delta > self.epsilon) and (itera < max_iter):
			delta = 0
			V0 = cp.deepcopy(V1)
			for s in range(self.nS):
				temp = 0
				if first_time: # At initialization of the policy iteration we use the stochastic uniform policy.
					p = 1 / self.nA
					delta = 100
					for a in range(self.nA):
						for next_s in range(self.nS):
							V1[s] += self.transitions[s, a, next_s] * self.rewards[s, a]
						delta = max((delta, abs(V1[s] - V0[s])))
				else:
					a = self.policy[s]
					for next_s in range(self.nS):
						V1[s] += self.transitions[s, a, next_s] * (self.rewards[s, a] + V0[next_s])
					delta = max((delta, abs(V1[s] - V0[s])))
			itera += 1
		return V1
		
	
	def learn(self, max_iter = 100):
		V = self.value_it()
		self.update_policy(V)
		self.not_converged = True
		itera = 0
		while self.not_converged:
			if itera == 0:
				V = self.value_it(first_time = True)
			elif itera > max_iter:
				break
			else:
				V = self.value_it()
			self.update_policy(V) # It also update self.not_converged (if the policy doesn't change then not_converged = False)
			itera += 1
		

			
class Opti_77_4room:
	def __init__(self,env):
		self.env=env
		pol = (
			[[0, 0, 0, 0, 0, 0, 0],
			[0, 1, 3, 3, 1, 1, 0],
			[0, 1, 2, 0, 1, 2, 0],
			[0, 1, 0, 0, 1, 0, 0],
			[0, 3, 3, 3, 3, 1, 0],
			[0, 0, 0, 0, 3, 1, 0],
			[0, 0, 0, 0, 0, 0, 0]]
		)
		self.policy = np.zeros(49)
		for x in range(7):
			for y in range(7):
				self.policy[x * 7 + y] = pol[x][y]
		self.mapping = env.mapping

	def name(self):
		return "Opti_77_4room"

	def reset(self,inistate):
		()

	def play(self,state):
		s = self.mapping[state]
		return self.policy[s]

	def update(self, state, action, reward, observation):
		()
		
		
class Opti_911_2room:
	def __init__(self,env):
		self.env=env
		pol = (
			[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
			[0, 3, 3, 3, 3, 1, 2, 2, 2, 2, 0],
			[0, 3, 3, 3, 3, 1, 2, 2, 2, 2, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 0],
			[0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0],
			[0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
		)
		self.policy = np.zeros(9*11)
		for x in range(9):
			for y in range(11):
				self.policy[x * 11 + y] = pol[x][y]
		self.mapping = env.mapping

	def name(self):
		return "Opti_911_2room"

	def reset(self,inistate):
		()

	def play(self,state):
		s = self.mapping[state]
		return self.policy[s]

	def update(self, state, action, reward, observation):
		()

