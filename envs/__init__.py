#!/usr/bin/env python

from gym.envs.registration import register


register(
    id='Continuous2D-v0',
    entry_point='envs.Continuous2D:Continuous2DEnv',
)


