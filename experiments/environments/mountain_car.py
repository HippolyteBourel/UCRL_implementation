# This file provide a discretized version of the mountain-car problem (it is based on the mountain-car of OpenAI's gym package).
# see: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

# Discretized using the wrapper method discretize(postion, velocity)
class MountainCarEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}

	def __init__(self):
		self.min_position = -1.2
		self.max_position = 0.6
		self.max_speed = 0.07
		self.goal_position = 0.5
		
		self.force=0.001
		self.gravity=0.0025

		self.low = np.array([self.min_position, -self.max_speed])
		self.high = np.array([self.max_position, self.max_speed])

		self.viewer = None

		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def disretize_p(self, position):
		if position < -0.9:
			return 0
		elif position < -0.6:
			return 1
		elif position < -0.4:
			return 2
		elif position < -0.2:
			return 3
		elif position < 0:
			return 4
		elif position < 0.2:
			return 5
		elif position < 0.4:
			return 6
		elif position < 0.5:
			return 7
		else:
			return 8

	# Discretization over 135 states of the state space (9 positions and 15 velocity)
	def discretize(self, position, velocity):
		d_pos = self.discretize_p(position)
		d_vel = int(velocity * 100) + 7
		state = d_pos * 15 + d_vel
		return state

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		position, velocity = self.state
		velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
		velocity = np.clip(velocity, -self.max_speed, self.max_speed)
		position += velocity
		position = np.clip(position, self.min_position, self.max_position)
		if (position==self.min_position and velocity<0): velocity = 0

		done = False#bool(position >= self.goal_position)
		if position >= self.goal_position:
			reward = 1
		else:
			reward = 0#-0.01

		self.state = (position, velocity)
		return self.dicretize(position, velocity), reward, done, {}

	def reset(self):
		self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return self.discretize(self.state[0], self.state[1])

	def _height(self, xs):
		return np.sin(3 * xs)*.45+.55

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.max_position - self.min_position
		scale = screen_width/world_width
		carwidth=40
		carheight=20


		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			xs = np.linspace(self.min_position, self.max_position, 100)
			ys = self._height(xs)
			xys = list(zip((xs-self.min_position)*scale, ys*scale))

			self.track = rendering.make_polyline(xys)
			self.track.set_linewidth(4)
			self.viewer.add_geom(self.track)

			clearance = 10

			l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
			car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			car.add_attr(rendering.Transform(translation=(0, clearance)))
			self.cartrans = rendering.Transform()
			car.add_attr(self.cartrans)
			self.viewer.add_geom(car)
			frontwheel = rendering.make_circle(carheight/2.5)
			frontwheel.set_color(.5, .5, .5)
			frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
			frontwheel.add_attr(self.cartrans)
			self.viewer.add_geom(frontwheel)
			backwheel = rendering.make_circle(carheight/2.5)
			backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
			backwheel.add_attr(self.cartrans)
			backwheel.set_color(.5, .5, .5)
			self.viewer.add_geom(backwheel)
			flagx = (self.goal_position-self.min_position)*scale
			flagy1 = self._height(self.goal_position)*scale
			flagy2 = flagy1 + 50
			flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
			self.viewer.add_geom(flagpole)
			flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
			flag.set_color(.8,.8,0)
			self.viewer.add_geom(flag)

		pos = self.state[0]
		self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
		self.cartrans.set_rotation(math.cos(3 * pos))

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
	
	def get_keys_to_action(self):
		return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys 
	
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None