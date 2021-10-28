from re import purge
import sys, os, time, random, argparse
from matplotlib.pyplot import draw
from pygame.constants import KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, K_w
from ruamel.yaml import YAML
import numpy as np
import gym
import pygame
import envs
import datetime

yaml = YAML()
config = yaml.load(open(sys.argv[1]))

env_name = config['env_name']
env      = gym.make(env_name, config=config, reward_func=None)
num_expert = config['num_traj']
x_d = config['goal_state']


flower1 = pygame.image.load('envs/media/sakura.png')
flower1 = pygame.transform.rotozoom(flower1, 0, 0.1)

flower2 = pygame.image.load('envs/media/flower.png')
flower2 = pygame.transform.rotozoom(flower2, 0, 0.1)

perfume1 = pygame.image.load('envs/media/perfume.png')
perfume1 = pygame.transform.rotozoom(perfume1, 0, 0.1)

perfume2 = pygame.image.load('envs/media/fragance.png')
perfume2 = pygame.transform.rotozoom(perfume2, 0, 0.1)

trash = pygame.image.load('envs/media/trash-can.png')
trash = pygame.transform.rotozoom(trash, 0, 0.1)


trajs = []
for n in range(num_expert):
    pygame.init()
    width = 1000
    height = 500
    screen = pygame.display.set_mode([width, height])
    screen.fill((255,255,255))
    clock = pygame.time.Clock()
    pygame.joystick.init()

    done = False
    state = env.reset()

    traj = [state]

    while not done:

        
        screen.fill((255,255,255))
        #for event in pygame.event.get():
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.JOYBUTTONUP:
            print("button released")

        joystick_count = pygame.joystick.get_count()
        
        joystick = pygame.joystick.Joystick(0)
        joystick.init()


        lr_axis = joystick.get_axis(0)*0.5
        #print(0, lr_axis)

        ud_axis = -joystick.get_axis(1)*0.5
        #print(1, ud_axis)

        if abs(lr_axis) > 1e-3 or abs(ud_axis) > 1e-3:
            state,_,done, _ = env.step(np.array([lr_axis, ud_axis]))
            print(state)
            traj.append(state)

        pygame.draw.circle(screen, (255,0,0), (env.mid_state1[0]*width, height - (env.mid_state1[1]/0.5)*height), (env.mid_state_r/0.5)*500)
        pygame.draw.circle(screen, (255,255,0), (env.left_state1[0]*width, height - (env.left_state1[1]/0.5)*height), (env.left_state_r/0.5)*500)
        pygame.draw.circle(screen, (0,255,255), (env.right_state1[0]*width, height - (env.right_state1[1]/0.5)*height), (env.right_state_r/0.5)*500)

        pygame.draw.circle(screen, (170,170,0), (env.left_state2[0]*width, height - (env.left_state2[1]/0.5)*height), (env.left_state_r2/0.5)*500)
        pygame.draw.circle(screen, (250,30,0), (env.right_state2[0]*width, height - (env.right_state2[1]/0.5)*height), (env.right_state_r2/0.5)*500)

        pygame.draw.circle(screen, (0,255,0), (x_d[0]*width, height - (x_d[1]/0.5)*height), 10)
        pygame.draw.circle(screen, (0,0,255), (state[0]*width, height - (state[1]/0.5)*height), 10)

        screen.blit(flower1, (env.left_state1[0]*width - 25, height - (env.left_state1[1]/0.5)*height - 25))
        screen.blit(perfume1, (env.mid_state1[0]*width - 25, height - (env.mid_state1[1]/0.5)*height - 25))
        screen.blit(trash, (env.right_state1[0]*width - 25, height - (env.right_state1[1]/0.5)*height - 25))
        screen.blit(flower2, (env.left_state2[0]*width - 25, height - (env.left_state2[1]/0.5)*height - 25))
        screen.blit(perfume2, (env.right_state2[0]*width - 25, height - (env.right_state2[1]/0.5)*height - 25))


        pygame.display.update()
        clock.tick(20)



    fill_arr = np.tile(state, (env.max_step - len(traj), 1))
    traj = np.array(traj)

    final_traj = np.vstack((traj, fill_arr))
    print(final_traj.shape)
    trajs.append(final_traj)

    pygame.quit()

human_trajs = trajs

# max_len = max([len(i) for i in human_trajs])
# for tr in range(len(human_trajs)):
#     mod_x = x_d.copy()
#     mod_x[1] = 1.0-mod_x[1]
#     fill_arr = np.tile(mod_x*500, (max_len - len(human_trajs[tr]), 1))
#     traj = np.vstack((human_trajs[tr], fill_arr))
#     human_trajs[tr] = np.array(traj)




# human_trajs = np.array(human_trajs)/500.0
# human_trajs[:, :, 1] = 1.0 - human_trajs[:, :, 1]
# human_trajs[:, 0, :] = state

print(human_trajs[0])

print(env.get_feature_expectation(human_trajs[0]))


np.save("expert_trajs/{}/expert_trajs_{}_{}.npy".format(env_name, env_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), human_trajs)
