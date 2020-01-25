# system imports
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# log videos
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import cv2

# algorithm imports
from agent import ppo_update, compute_gae
from model import ActorCritic
from multiprocessing_env import SubprocVecEnv
import utils
from utils import print_

# environment imports
import pybullet as p
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))

def log_video(frames):
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	for frame in frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out.write(frame)
	out.release()

def running_test(log_dir, max_steps=100, create_render=False):
# Parameters
	hidden_size = [256,256]
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")


	# Create test environment.
	env = gym.make('LunarLander-v2')

	# Check availability of cuda
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# State space and action space
	num_inputs = env.observation_space.shape[0]
	num_outputs = env.action_space.n 	#discrete action space

	# Create network/policy.
	net = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)

	checkpoint = torch.load(os.path.join(log_dir,'models/weights_49000.pth'), map_location='cpu')
	net.load_state_dict(checkpoint['model'])

	state = env.reset()

	
	# if create_video: frames.append(img)
	done = False
	total_reward = 0
	steps = 0
	print_('Test Started...', color='r', style='bold')


	while not done and steps < max_steps:
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		probabilities, value = (net(state))
			# print("#############",probabilities)
			# action = dist.sample()
			

		action_probs = torch.distributions.Categorical(probabilities.cpu())
		action = action_probs.sample()
		action_ = action.detach().numpy()
		# print(dist.sample().cpu().numpy()[0])
		print("Action is ",int(action_))
		next_state, reward, done, info = env.step(int(action_)) #Hack
		if create_render: env.render()

		# print(info.keys())
		# not sure
		# STATES = STATES + info['internal_observations']
		# if create_video: frames = frames + info['frames']
		# link_positions = link_positions + info['link_positions']

		print("Step No: {:3d}, Reward: {:2.3f} and done: {}".format(steps, reward, done))
		state = next_state
		total_reward += reward
		steps += 1
		# STATES.append(state)
	print_('Total Reward: {}'.format(total_reward), color='bl', style='bold')
	print('Test Ended!')
	# print(len(link_positions))
	# np.savetxt('rl_policy_linkpositions_2.txt', np.array(link_positions))
	# return STATES

# state = running_test()
# print(state)
# print(type(state))

if __name__ == "__main__":
	log_dir = 'log_24_01_2020_15_50_08'
	max_steps = 2000
	create_render = True
	state = running_test(log_dir, max_steps=max_steps, create_render=create_render)