import gym
import numpy as np
import argparse
import pybullet_envs
from gym import wrappers
import os
import sys
import time

def create_env(env_name):
	env = gym.make(env_name)
	return env

def softmax(inputs):
	return np.exp(inputs) / float(sum(np.exp(inputs)))

def policy(state, weights):
	# state: 	MountainCar (2,1)
	# weights:	MountainCar (1,2)
	probabilities = np.matmul(weights, state.reshape(-1,1))
	return np.argmax(softmax(probabilities))

def test_env(env, policy, weights, normalizer=None, path=None):
	# Argument:
		# env:			Object of the gym environment.
		# policy:		A function that will take weights, state and returns actions
	if path: np.savetxt(path+'.txt', weights)		
	state = env.reset()
	done = False
	total_reward = 0.0
	total_states = []
	steps = 0

	while not done and steps<500:
		if normalizer:
			state = normalizer.normalize(state)
		action = policy(state, weights)
		next_state, reward, done, _ = env.step(action)
		#if abs(next_state[2]) < 0.0001*10:
		#	reward = -100
		#	done = True
		print(reward, done, action)
		# reward = max(min(reward, 1), -1)
		env.render()
		time.sleep(0.1)

		total_states.append(state)
		total_reward += reward
		steps += 1
		state = next_state
	env.close()
	print(float(total_reward), steps)

#################### Normalizing the states #################### 
class Normalizer():
	def __init__(self, nb_inputs):
		self.mean = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def restore(self, path):
		self.mean = np.loadtxt(os.path.join(path, 'mean.txt'))
		self.var = np.loadtxt(os.path.join(path, 'var.txt'))

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std

if __name__ == '__main__':
	idx = int(sys.argv[1])
	path = os.path.join('exp_lunarLander', 'models', 'policy'+str(idx))
	weights = np.loadtxt(os.path.join(path, 'weights.txt'))
	env = create_env('LunarLander-v2')
	# env = wrappers.Monitor(env, 'videos', force=True)
	normalizer = Normalizer([1, env.observation_space.shape[0]])
	normalizer.restore(path)

	test_env(env, policy, weights, normalizer=normalizer)