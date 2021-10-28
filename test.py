import sys, os, time, random, argparse
import numpy as np
import matplotlib.pyplot as plt
from controller import LQR
#from env import DoubleIntegrator_2DPointMassEnv
import envs
import torch
from reward_net import SimpleReward
import gym
from gym import spaces, logger
from agent import Agent
import loss_fn
import param_store
import datetime
from ruamel.yaml import YAML

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")


yaml = YAML()
config = yaml.load(open(sys.argv[1]))

env_name = config['env_name']
checkpoint_fpath = "{}/{}".format(env_name, config['test_model_name'] )
checkpoint = torch.load("trained_models/"+checkpoint_fpath + '.pt')




reward_func = SimpleReward(6, 64).to(device)
reward_optim = torch.optim.Adam(reward_func.parameters(), lr=0.001)


reward_func.load_state_dict(checkpoint['reward_func'])
reward_optim.load_state_dict(checkpoint['reward_optim'])

env = gym.make(env_name, config=config, reward_func=reward_func)
env.visual_reward(show=True, path=None)

#for plotting
# file_name = config['expert_controller']['traj_file']
# human_trajs = np.load("expert_trajs/{}/{}.npy".format(env_name, file_name))
# expert_trajs = np.array(human_trajs).copy()
# single_traj = expert_trajs[0]
# env.visual_traj_reward(single_traj)



agent = Agent(gym.make(env_name, config=config, reward_func=reward_func), 50000, 1)
test_traj = np.array(agent.generate_agent_traj(1))[0]


print(test_traj.shape)


# fig, axs = plt.subplots(3,1, gridspec_kw={'height_ratios': [6, 1, 1]}, figsize=(16,8))
# single_test_traj = test_traj


# #axs[0].plot(range(len(single_test_traj[:,0])),single_test_traj[:,0], color='red')

# axs[0].plot(single_test_traj[:,0], single_test_traj[:,1])
# axs[1].plot(np.linspace(0,len(single_test_traj), len(single_test_traj)), single_test_traj[:,0])
# axs[2].plot(np.linspace(0,len(single_test_traj), len(single_test_traj)), single_test_traj[:,1])
# axs[0].set_ylim(env.observation_space.low[1], env.observation_space.high[1])
# axs[0].set_xlim(env.observation_space.low[0], env.observation_space.high[0])

# plt.show()
env.visual_traj_reward(test_traj)



