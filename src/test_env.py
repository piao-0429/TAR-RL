import os, psutil
# make sure mujoco and nvidia will be found
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
                                ':/home/hp/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/home/hp/.mujoco/mujoco210/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import shutil
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'
import platform
os.environ['MUJOCO_GL'] = 'egl'
# set to glfw if trying to render locally with a monitor
# os.environ['MUJOCO_GL'] = 'glfw'
os.environ['EGL_DEVICE_ID'] = '0'
from distutils.dir_util import copy_tree
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import imageio

import hydra
import torch
from dm_env import StepType, TimeStep, specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from traj_buffer import TrajBufferStorage, make_traj_loader
from video import TrainVideoRecorder, VideoRecorder
import joblib
import pickle
import time

import envs.dmc.dmc as dmc
import numpy as np

env_names = ["walker_walk_1", "walker_walk_2", "walker_walk_3", "walker_walk_4", "walker_walk_5", "walker_walk_6"]

env_list, _ = dmc.make_env(env_names, "rescale", 3,2,1)

env = env_list[0]

action_spec = env.action_spec()

action = np.zeros(action_spec.shape, dtype=np.float32)

action[0] = 1.2

action[1] = 0.5

env.reset()
time_step = env.step(action)

print(time_step)