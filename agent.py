
import itertools
from stable_baselines3 import SAC, PPO, DDPG, DQN, HerReplayBuffer
import numpy as np
from torch.nn.functional import normalize

class Agent:
    def __init__(self, env, total_timesteps, log_interval):
        self.env = env

        self.model = SAC('MlpPolicy', env, verbose=1)
        #self.model = PPO('MlpPolicy', env, learning_rate=1e-4, n_steps=1024, gamma=0.9, gae_lambda=0.95, verbose=1)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)#5000 for di 20000 for lane

    def generate_agent_traj(self, n_traj):
        trajs = []
        
        for i in range(n_traj):
            
            obs = self.env.reset()
            
            single_traj = []
            done = False

            t = 0
            while not done:
                single_traj.append(obs)
                action, _states = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.env.step(action)

                t += 1
                
            single_traj.append(obs)
            trajs.append(single_traj)

        return np.array(trajs)

    # def generate_test_traj(self, n_traj):
        
    #     trajs = []

    #     acts = []
        
        
    #     for i in range(n_traj):
            
    #         obs = self.env.reset()

            
    #         single_traj = []
    #         single_act = []
    #         done = False

    #         while not done:
    #             single_traj.append(self.env.state)
                
    #             action, _states = self.model.predict(obs, deterministic=True)
    #             single_act.append(action)
    #             obs, _, done, _ = self.env.step(action)
                
    #         single_traj.append(self.env.state)
    #         trajs.append(single_traj)
    #         acts.append(single_act)

    #     return np.array(trajs), np.array(acts)

