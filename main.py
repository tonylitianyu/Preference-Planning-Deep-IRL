import sys, os, time, random, argparse
from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
from controller import LQR, MPC
#from env import DoubleIntegrator_2DPointMassEnv
import envs
import torch
# from reward_net import MLPReward
import gym
from gym import spaces, logger

from reward_net import SimpleReward
from agent import Agent
import loss_fn
import param_store
import datetime
np.set_printoptions(threshold=sys.maxsize)



yaml = YAML()
config = yaml.load(open(sys.argv[1]))

env_name = config['env_name']
env      = gym.make(env_name, config=config, reward_func=None)

controller_name = config['expert_controller']['controller_type']
expert_trajs = []


if controller_name == 'Human':
    file_name = config['expert_controller']['traj_file']
    human_trajs = np.load("expert_trajs/{}/{}.npy".format(env_name, file_name))
    expert_trajs = np.array(human_trajs).copy()
else:

    if controller_name == 'LQR':
        policy = LQR(env)

    num_expert = config['num_traj']
    num_learner = config['num_learner']
    expert_traj_len = config['expert_traj_len']
    learner_traj_len = expert_traj_len

    for et in range(num_expert):
        single_traj = []
        u_list = []
        mu_list = []
        state = env.reset()
        done = False
        while not done:
            a = policy(state, env.goal_state)
            a = np.clip(a, env.action_space.low, env.action_space.high)

            single_traj.append(state)
            u_list.append(a)
            state, _, done , _ = env.step(a)

        single_traj= np.array(single_traj).squeeze() #(step, n_state)
        expert_trajs.append(single_traj)



for i in range(len(expert_trajs)):
    single_traj = expert_trajs[i]
    expert_fe_traj = env.get_feature_expectation(single_traj)
    print(expert_fe_traj)
    env.visual_expert(single_traj)



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# agent = Agent(gym.make(env_name, config=config, reward_func=None), 10000, 1)
# test_traj = agent.generate_agent_traj(1)[0]
# plt.plot(test_traj[:,0],test_traj[:,1], color='red')
# plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
# plt.xlim(env.observation_space.low[0], env.observation_space.high[0])

# plt.show()


reward_func = SimpleReward(6, 64).to(device)
reward_optim = torch.optim.Adam(reward_func.parameters(), lr=0.001)#)

curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
curr_run_path = "trained_models/{}/{}".format(env_name, curr_time)
if not os.path.isdir(curr_run_path):
   os.makedirs(curr_run_path)

curr_model_path = curr_run_path + "/models"
if not os.path.isdir(curr_model_path):
   os.makedirs(curr_model_path)

curr_visual_path = curr_run_path + "/visuals"
if not os.path.isdir(curr_visual_path):
   os.makedirs(curr_visual_path)


for i_epi in range(config['reward_epi']):
    #train reward
    agent = Agent(gym.make(env_name, config=config, reward_func=reward_func), 30000, 30000)
    test_traj = agent.generate_agent_traj(1)[0]  #(10,100,4)
    
    print("very last state: ", test_traj[-1])
    curr_policy_fe_traj = env.get_feature_expectation(test_traj)
    loss = loss_fn.maxentirl_loss(curr_policy_fe_traj, expert_fe_traj, reward_func, device)

    print("Episode {} Loss: {}".format(i_epi, loss.item()))

    reward_optim.zero_grad()
    loss.backward()
    reward_optim.step()

    if i_epi % 1 == 0:
        #save reward model
        reward_checkpoint = {
            'reward_func' : reward_func.state_dict(),
            'reward_optim' : reward_optim.state_dict(),
        }
        param_store.save_checkpoint(reward_checkpoint, "{}/{}/models/{}.pt".format(env_name, curr_time,i_epi))


        env = gym.make(env_name, config=config, reward_func=reward_func)
        fig_name = "trained_models/{}/{}/visuals/{}.png".format(env_name, curr_time,i_epi)
        env.visual_reward(show=False, path=fig_name)


