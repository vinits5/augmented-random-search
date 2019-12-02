import gym
import numpy as np
import argparse
import pybullet_envs
from gym import wrappers
import os
from tensorboardX import SummaryWriter

#################### Environment and Agent ####################
def create_env(env_name):
	env = gym.make(env_name)
	return env

def policy(state, weights):
	# state: 	MountainCar (2,1)
	# weights:	MountainCar (1,2)
	return np.matmul(weights, state.reshape(-1,1))#.reshape(-1,1)

def test_env(env, policy, weights, normalizer=None):
	# Argument:
		# env:			Object of the gym environment.
		# policy:		A function that will take weights, state and returns actions
	state = env.reset()
	done = False
	total_reward = 0.0
	total_states = []
	steps = 0

	while not done and steps<1000:
		if normalizer:
			normalizer.observe(state)
			state = normalizer.normalize(state)
		action = policy(state, weights)
		next_state, reward, done, _ = env.step(action)

		total_states.append(state)
		total_reward += reward
		steps += 1
		state = next_state

	return total_reward

def log_video(env, args, policy, weights, path, normalizer=None):
	# env = create_env(args.env)

	
	nb_inputs = env.observation_space.shape[0]
	nb_outputs = env.action_space.shape[0]
	
	state = env.reset()
	done = False
	num_plays = 0.
	sum_rewards = 0
	while not done and num_plays < 1000:
		if normalizer:
			normalizer.observe(state)
			state = normalizer.normalize(state)
		action = policy(state, weights)
		state, reward, done, _ = env.step(action)
		sum_rewards += reward
		num_plays += 1
	return sum_rewards, num_plays


#################### ARS algorithm ####################
def sort_directions(data, b):
	reward_p, reward_n = data
	reward_max = []
	for rp, rn in zip(reward_p, reward_n):
		reward_max.append(max(rp, rn))

	# ipdb.set_trace()
	idx = np.argsort(reward_max)	# Sort rewards and get indices.
	idx = np.flip(idx)				# Flip to get descending order.

	return idx

def update_weights(data, lr, b, weights):
	reward_p, reward_n, delta = data
	idx = sort_directions([reward_p, reward_n], b)

	step = np.zeros(weights.shape)
	for i in range(b):
		step += [reward_p[idx[i]] - reward_n[idx[i]]]*delta[idx[i]]

	sigmaR = np.std(np.array(reward_p)[idx][:b] + np.array(reward_n)[idx][:b])
	weights += (lr*1.0)/(b*sigmaR*1.0)*step

	return weights

def sample_delta(size):
	return np.random.normal(size=size)


#################### Normalizing the states #################### 
class Normalizer():
	def __init__(self, nb_inputs):
		self.n = np.zeros(nb_inputs)
		self.mean = np.zeros(nb_inputs)
		self.mean_diff = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def observe(self, x):
		self.n += 1.
		last_mean = self.mean.copy()
		self.mean += (x - self.mean) / self.n
		self.mean_diff += (x - last_mean) * (x - self.mean)
		self.var = (self.mean_diff / self.n).clip(min=1e-2)

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std

#################### Training ARS Class #################### 
class ARS:
	def __init__(self, args):
		self.v = args.v
		self.N = args.N
		self.b = args.b
		self.lr = args.lr
		self.env = create_env(args.env)
		self.env = wrappers.Monitor(self.env, 'exp', force=True)
		self.args = args

		if not os.path.exists(args.log): os.mkdir(args.log)

		# For MountainCar -> (1,2)
		self.size = [self.env.action_space.shape[0], self.env.observation_space.shape[0]]
		self.weights = np.zeros(self.size)
		self.threshold_reward = args.threshold_reward
		if args.normalizer: self.normalizer = Normalizer([1,self.size[1]])
		else: self.normalizer=None

	def train_one_epoch(self):
		delta = [sample_delta(self.size) for _ in range(self.N)]

		reward_p = [test_env(self.env, policy, self.weights + self.v*x, normalizer=self.normalizer) for x in delta]
		reward_n = [test_env(self.env, policy, self.weights - self.v*x, normalizer=self.normalizer) for x in delta]
		
		return update_weights([reward_p, reward_n, delta], self.lr, self.b, self.weights)

	def train(self):
		writer = SummaryWriter('cheetah')
		# test_reward = test_env(self.env, policy, self.weights, normalizer=self.normalizer)
		print('Training Begins!')
		counter = 0
		# while test_reward < self.threshold_reward:
		while counter < 1000:
			print('Counter: {}'.format(counter))
			self.weights = self.train_one_epoch()

			path = os.path.join(self.args.log, 'exp'+str(counter))
			test_reward, num_plays = log_video(self.env, self.args, policy, self.weights, path, normalizer=self.normalizer)
			writer.add_scalar('test_reward', test_reward, counter)
			writer.add_scalar('episodic_steps', num_plays, counter)
			print('Iteration: {} and Reward: {}'.format(counter, test_reward))
			counter += 1

		counter = 0		
		while True:
			path = os.path.join(self.args.log, 'test'+str(counter))
			print(test_env(self.args, policy, self.weights, path, normalizer=self.normalizer))
			counter += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='ARS Parameters')
	parser.add_argument('--v', type=float, default=0.03, help='noise in delta')
	parser.add_argument('--N', type=int, default=30, help='No of perturbations')
	parser.add_argument('--b', type=int, default=16, help='No of top performing directions')
	parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
	parser.add_argument('--threshold_reward', type=float, default=92.0, help='threshold_reward')
	parser.add_argument('--normalizer', type=bool, default=False, help='use normalizer')
	parser.add_argument('--env', type=str, default='HalfCheetahBulletEnv-v0', help='name of environment')
	parser.add_argument('--log', type=str, default='exp', help='Log folder to store videos')

	args = parser.parse_args()
	ars = ARS(args)
	ars.train()