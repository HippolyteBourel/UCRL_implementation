from learners.UCRL2_L import *
import scipy as sp
import numpy as np

# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
class UCRL2_L_sqrtSC(UCRL2_L):
	def name(self):
		return "UCRL2_L_sqrtSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[state, action] >= np.sqrt(max([1, self.Nk[state, action]])): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action