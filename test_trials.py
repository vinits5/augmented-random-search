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

def policy(state, weights):
	return np.matmul(weights, state.reshape(-1,1))#.reshape(-1,1)

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

	while not done and steps<5000:
		if normalizer:
			state = normalizer.normalize(state)
		action = policy(state, weights)
		next_state, reward, done, _ = env.step(action)

		total_states.append(state)
		total_reward += reward
		steps += 1
		state = next_state
	print(float(total_reward), steps)
	if path is None: return float(total_reward)
	else: return float(total_reward), steps

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
	#idx = int(sys.argv[1])
	idx = 910
	path = os.path.join('results', 'policy'+str(idx))
	weights = np.loadtxt(os.path.join(path, 'weights.txt'))
	env = create_env('BipedalWalker-v2')
	normalizer = Normalizer([1, env.observation_space.shape[0]])
	normalizer.restore(path)


	rewards = [test_env(env, policy, weights, normalizer=normalizer) for _ in range(100)]
	np.savetxt('avg_reward.txt', rewards)
	print("Mean of 100 trials: ",sum(rewards)/100.0)

	rewards = np.array(rewards).reshape(1,-1)

	import matplotlib.pyplot as plt
	print('Mean of 100 trials: ', np.mean(rewards))
	plt.plot(rewards[0], linewidth=3)
	plt.title('BipedalWalker-v2 (ARS Tests)', fontsize=30)
	plt.xlabel('No of Trials', fontsize=30)
	plt.ylabel('Reward per Trial', fontsize=30)
	plt.tick_params(labelsize=20, width=3, length=10)
	plt.axhline(y=np.mean(rewards), linestyle='--', linewidth=3, color='r')
	plt.show()
