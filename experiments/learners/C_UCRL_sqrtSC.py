from learners.C_UCRL import *
import scipy as sp
import numpy as np
import copy as cp

# Work in progress.

# C_UCRL is the algorithm introduced by Maillard and Asadi 2018, it is the more realistic version were C (the classes) and sigma (the profile
# mapping) are unknown. It extends C_UCRL_C which is the implementation of the algorihtm C_UCRL(C) of the paper, using clustering in order to
# estimate C.
class C_UCRL_sqrtSC(C_UCRL):
	def name(self):
		return "C_UCRL_sqrtSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[self.C[state, action]] >= np.sqrt(max([1, self.Nk_c[self.C[state, action]]])): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action