import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian, grad
import random
import math
import gym
from gym import spaces, logger
import torch
import matplotlib.pyplot as plt
import matplotlib.image as image
np.set_printoptions(suppress=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class Continuous2DEnv(gym.Env): 
    def __init__(self, config, reward_func): 
        super(Continuous2DEnv, self).__init__()
        self.config = config
        self.dt = 0.1
        self.num_states = 2
        self.num_actions = 2

        self.space_size = config['width']
        self.goal_state = np.array(config['goal_state'])
        self.start_state = np.array(config['start_state'])

        self._A = np.array([[0,0],
                            [0,0]])
        self._B = np.array([[1,0],
                            [0,1]]) 

        max_action = np.array([0.5,0.5])
        self.action_space = spaces.Box(low=-max_action,high=max_action,shape=(2,))

        high = np.array([1, 0.5])

        low = np.array([0,0])

        self.observation_space = spaces.Box(low, high)
        
        self.reward_func = reward_func

        self.Q = np.diag([1,1]) 
        self.R = np.diag([10,10])

        self.max_step = 1000
        self.curr_step = 0


        self.mid_state1 = np.array([0.3, 3*0.0833])
        self.mid_state_r = 0.0833

        self.right_state1 = np.array([0.3, 0.0833])
        self.right_state_r = 0.0833

        self.left_state1 = np.array([0.3, 5*0.0833])
        self.left_state_r = 0.0833



        self.left_state2 = np.array([0.7, 3*0.125])
        self.left_state_r2 = 0.125

        self.right_state2 = np.array([0.7, 0.125])
        self.right_state_r2 = 0.125

        
    def get_lin(self,x=np.zeros(4),u=np.zeros(2)): 
        return self._A, self._B

    def reset(self,state=None):

        self.state = self.start_state
        self.curr_step = 0

        return self.state.copy()

    def get_obs_airl(self, state):
        dis = np.linalg.norm(state - self.goal_state)#/np.square(np.linalg.norm(self.observation_space.low - self.goal_state))


        mid_dis1 = 0.0
        left_dis1 = 0.0
        right_dis1 = 0.0
        
        left_dis2 = 0.0
        right_dis2 = 0.0

        if np.linalg.norm(state - self.mid_state1) < self.mid_state_r:
            mid_dis1 = 1.0#np.linalg.norm(state - self.mid_state)#/np.square(np.linalg.norm(np.array([1.0,0.0]) - self.mid_state))

        if np.linalg.norm(state - self.right_state1) < self.right_state_r:
            right_dis1 = 1.0#np.linalg.norm(state - self.right_state)
        
        if np.linalg.norm(state - self.left_state1) < self.left_state_r:
            left_dis1 = 1.0#np.linalg.norm(state - self.left_state)

        if np.linalg.norm(state - self.left_state2) < self.left_state_r2:
            left_dis2 = 1.0#np.linalg.norm(state - self.mid_state)#/np.square(np.linalg.norm(np.array([1.0,0.0]) - self.mid_state))

        if np.linalg.norm(state - self.right_state2) < self.right_state_r2:
            right_dis2 = 1.0#np.linalg.norm(state - self.right_state)
        

        return np.array([dis, mid_dis1, left_dis1, right_dis1, left_dis2, right_dis2])

    def get_feature_expectation(self, traj):
        feature_traj = []
        for t in range(0,len(traj)):
            curr_obs = (self.config['feature_gamma']**t)*self.get_obs_airl(traj[t])
            feature_traj.append(curr_obs)
        return np.array(feature_traj)

    
    def f(self,x,u): # transiton model
        return self._A @ x + self._B @ u 
    
    def step(self,u):
        self.curr_step += 1

        self.state = self.state + self.f(self.state.reshape(-1,1),u.reshape(-1,1)).flatten() * self.dt
        reward = 0.0

        
        done = False
        self.state[0] = np.clip(self.state[0],self.observation_space.low[0],self.observation_space.high[0])
        self.state[1] = np.clip(self.state[1],self.observation_space.low[1],self.observation_space.high[1])


        if self.reward_func is not None:
            fe_torch = torch.FloatTensor(self.get_obs_airl(self.state)).to(device)
            reward = self.reward_func.r(fe_torch).item()

        # if np.linalg.norm(self.state - self.goal_state) < 0.05:
        #     # reward += 1000.0
        #     done = True

        if self.curr_step >= self.max_step:
            done = True


        return self.state.copy(), reward, done, {}

    def visual_reward(self, show=False, path=None):
        xx = np.linspace(self.observation_space.low[0],self.observation_space.high[0],100)
        yy = np.linspace(self.observation_space.low[1],self.observation_space.high[1],50)
        X, Y = np.meshgrid(xx, yy)
        print(X.shape)
        print(Y.shape)

        xv_1d = np.expand_dims(X.flatten(), axis=1)
        yv_1d = np.expand_dims(Y.flatten(), axis=1)
        s = np.hstack((xv_1d, yv_1d))
        #s = np.hstack((xv_1d, np.zeros((len(s), 1)), yv_1d))

        reward_arr = []
        for state in s:
            state_torch = torch.FloatTensor(self.get_obs_airl(state)).to(device)
            reward = self.reward_func.r(state_torch).item()

            reward_arr.append(reward)

        reward_arr = np.array(reward_arr)

        if show:
            plt.figure(figsize=(11,5))
            plt.pcolormesh(X,Y,reward_arr.reshape(50, 100))
            plt.colorbar()
            plt.show()
        else:
            plt.figure(figsize=(11,5))
            plt.pcolormesh(X,Y,reward_arr.reshape(50, 100))
            plt.colorbar()
            plt.savefig(path)

    def visual_expert(self, traj):
        trash_im = image.imread('envs/media/trash-can.png')
        human_im = image.imread('envs/media/man-silhouette.png')
        sakura_im = image.imread('envs/media/sakura.png')
        perfume1_im = image.imread('envs/media/perfume.png')
        flower_im = image.imread('envs/media/flower.png')
        perfume2_im = image.imread('envs/media/fragance.png')
        flag_im = image.imread('envs/media/flag.png')
        img_size = 0.03

        def plot_img(axes, img, pos):
            axes.imshow(img, aspect='auto', extent=(pos[0]-img_size, pos[0]+img_size, pos[1]-img_size,pos[1]+img_size), zorder=10)

        fig, ax = plt.subplots(figsize=(12,6))
        
        plot_img(ax, trash_im, self.right_state1)
        plot_img(ax, human_im, self.start_state)
        plot_img(ax,sakura_im, self.left_state1)
        plot_img(ax, perfume1_im, self.mid_state1)
        plot_img(ax, flower_im, self.left_state2)
        plot_img(ax, perfume2_im, self.right_state2)
        plot_img(ax, flag_im, self.goal_state)


        
        ax.plot(traj[:,0],traj[:,1], color='red')
        ax.set_ylim(self.observation_space.low[1], self.observation_space.high[1])
        ax.set_xlim(self.observation_space.low[0], self.observation_space.high[0])
        

        plt.show()

    def visual_traj_reward(self, traj):

        xx = np.linspace(self.observation_space.low[0],self.observation_space.high[0],600)
        yy = np.linspace(self.observation_space.low[1],self.observation_space.high[1],300)
        X, Y = np.meshgrid(xx, yy)
        print(X.shape)
        print(Y.shape)

        xv_1d = np.expand_dims(X.flatten(), axis=1)
        yv_1d = np.expand_dims(Y.flatten(), axis=1)
        s = np.hstack((xv_1d, yv_1d))
        #s = np.hstack((xv_1d, np.zeros((len(s), 1)), yv_1d))

        reward_arr = []
        for state in s:
            state_torch = torch.FloatTensor(self.get_obs_airl(state)).to(device)
            reward = self.reward_func.r(state_torch).item()

            reward_arr.append(reward)

        reward_arr = np.array(reward_arr)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.pcolormesh(X,Y,reward_arr.reshape(300, 600))

        trash_im = image.imread('envs/media/trash-can.png')
        human_im = image.imread('envs/media/man-silhouette.png')
        sakura_im = image.imread('envs/media/sakura.png')
        perfume1_im = image.imread('envs/media/perfume.png')
        flower_im = image.imread('envs/media/flower.png')
        perfume2_im = image.imread('envs/media/fragance.png')
        flag_im = image.imread('envs/media/flag.png')
        img_size = 0.03

        def plot_img(axes, img, pos):
            axes.imshow(img, aspect='auto', extent=(pos[0]-img_size, pos[0]+img_size, pos[1]-img_size,pos[1]+img_size), zorder=10)
        
        plot_img(ax, trash_im, self.right_state1)
        plot_img(ax, human_im, self.start_state)
        plot_img(ax,sakura_im, self.left_state1)
        plot_img(ax, perfume1_im, self.mid_state1)
        plot_img(ax, flower_im, self.left_state2)
        plot_img(ax, perfume2_im, self.right_state2)
        plot_img(ax, flag_im, self.goal_state)


        
        ax.plot(traj[:,0],traj[:,1], color='red')
        ax.set_ylim(self.observation_space.low[1], self.observation_space.high[1])
        ax.set_xlim(self.observation_space.low[0], self.observation_space.high[0])
        

        plt.show()