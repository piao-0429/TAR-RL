# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
import wandb
import torch
from dm_env import StepType, TimeStep, specs

import utils
from logger_v2 import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from traj_buffer import TrajBufferStorage, make_traj_loader
from video import TrainVideoRecorder, VideoRecorder
import joblib
import pickle
import time


torch.backends.cudnn.benchmark = True

# TODO this part can be done during workspace setup
ENV_TYPE = 'dmc'
if ENV_TYPE == 'adroit':
    import mj_envs
    from mjrl.utils.gym_env import GymEnv
    from rrl_local.rrl_utils import make_basic_env, make_dir
    from adroit import AdroitEnv
else:
    import envs.dmc.dmc as dmc
IS_ADROIT = True if ENV_TYPE == 'adroit' else False

def get_ram_info():
    # get info on RAM usage
    d = dict(psutil.virtual_memory()._asdict())
    for key in d:
        if key != "percent": # convert to GB
            d[key] = int(d[key] / 1024**3 * 100)/ 100
    return d

def make_agent(data_spec_list, cfg, is_adroit):
    cfg.obs_shape = data_spec_list[0][0].shape
    if is_adroit:
        cfg.action_idx = np.cumsum([0] + [spec.shape[0] for spec in data_spec_list[0][2]])
    else:
        action_idx = np.cumsum([0] + [spec[1].shape[0] for spec in data_spec_list])
        cfg.action_idx = action_idx.tolist()
    cfg.action_shape = cfg.action_idx[-1]
        
    return hydra.utils.instantiate(cfg)

def print_stage1_time_est(time_used, curr_n_update, total_n_update):
    time_per_update = time_used / curr_n_update
    est_total_time = time_per_update * total_n_update
    est_time_remaining = est_total_time - time_used
    print("Stage 1 [{:.2f}%]. Frames:[{:.0f}/{:.0f}]K. Time:[{:.2f}/{:.2f}]hrs. Overall FPS: {}.".format(
        curr_n_update / total_n_update * 100, curr_n_update/1000, total_n_update/1000,
        time_used / 3600, est_total_time / 3600, int(curr_n_update / time_used)))

def print_stage2_time_est(time_used, curr_n_update, total_n_update):
    time_per_update = time_used / curr_n_update
    est_total_time = time_per_update * total_n_update
    est_time_remaining = est_total_time - time_used
    print("Stage 2 [{:.2f}%]. Frames:[{:.0f}/{:.0f}]K. Time:[{:.2f}/{:.2f}]hrs. Overall FPS: {}.".format(
        curr_n_update / total_n_update * 100, curr_n_update/1000, total_n_update/1000,
        time_used / 3600, est_total_time / 3600, int(curr_n_update / time_used)))

def print_stage3_time_est(time_used, curr_n_frames, total_n_frames):
    time_per_update = time_used / curr_n_frames
    est_total_time = time_per_update * total_n_frames
    est_time_remaining = est_total_time - time_used
    print("Stage 3 [{:.2f}%]. Frames:[{:.0f}/{:.0f}]K. Time:[{:.2f}/{:.2f}]hrs. Overall FPS: {}.".format(
        curr_n_frames / total_n_frames * 100, curr_n_frames/1000, total_n_frames/1000,
        time_used / 3600, est_total_time / 3600, int(curr_n_frames / time_used)))

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("\n=== Training log stored to: ===")
        print(f'workspace: {self.work_dir}')
        self.direct_folder_name = os.path.basename(self.work_dir)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.replay_buffer_fetch_every = 1000
        self.traj_buffer_fetch_every = 50
        if self.cfg.debug > 0: # if debug mode, then change hyperparameters for quick testing
            self.set_debug_hyperparameters()
        self.setup()

        self.agent = make_agent(self.data_specs_list,
                                self.cfg.agent, is_adroit=IS_ADROIT)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._env_steps = [0 for _ in range(len(self.train_env_list))]
        self._env_episodes = [0 for _ in range(len(self.train_env_list))]

    def set_debug_hyperparameters(self):
        self.cfg.num_seed_frames=1000 if self.cfg.num_seed_frames > 1000 else self.cfg.num_seed_frames
        self.cfg.agent.num_expl_steps=500 if self.cfg.agent.num_expl_steps > 1000 else self.cfg.agent.num_expl_steps
        if self.cfg.replay_buffer_num_workers > 1:
            self.cfg.replay_buffer_num_workers = 1
        self.cfg.num_eval_episodes = 1
        self.cfg.replay_buffer_size = 30000
        self.cfg.batch_size = 8
        self.cfg.feature_dim = 8
        self.cfg.num_train_frames = 5050
        self.replay_buffer_fetch_every = 30
        self.cfg.stage2_n_update = 100
        self.cfg.num_demo = 2
        self.cfg.eval_every_frames = 3000
        self.cfg.agent.hidden_dim = 8
        self.cfg.agent.num_expl_steps = 500
        self.cfg.stage2_eval_every_frames = 50
        
        self.cfg.traj_buffer_size = 8
        self.cfg.traj_batch_size = 4
        self.cfg.traj_buffer_num_workers = 1
        self.cfg.seq_len = 10
        self.traj_buffer_fetch_every = 2
        self.cfg.stage1_n_update = 10
        self.cfg.stage1_eval_every_frames = 5
        
        self.cfg.use_wb = False
        self.cfg.use_tb = False

    def setup(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        if self.cfg.save_models:
            assert self.cfg.action_repeat % 2 == 0

        # create logger
        env_names = self.cfg.task_names
        self.env_names = env_names
        self.loggers = []
        for i, env_name in enumerate(env_names):
            self.loggers.append(Logger(self.work_dir / (str(i)+'_'+env_name), use_wb=self.cfg.use_wb, use_tb=self.cfg.use_tb))
        self.loggers.append(Logger(self.work_dir, use_wb=self.cfg.use_wb, use_tb=self.cfg.use_tb))
        if self.cfg.use_wb:
            wandb.init(project=self.cfg.project, dir=str(self.work_dir), id=self.cfg.experiment+'-'+str(self.cfg.seed))
        # assert env_name in ('hammer-v0','door-v0','pen-v0','relocate-v0',)
        self.env_type = ENV_TYPE
        if self.cfg.agent.encoder_lr_scale == 'auto':
            # if env_name == 'relocate-v0':
            #     self.cfg.agent.encoder_lr_scale = 0.01
            # else:
            #     self.cfg.agent.encoder_lr_scale = 1
            self.cfg.agent.encoder_lr_scale = 1

        self.env_feature_type = self.cfg.env_feature_type
        
        self.train_env_list = []
        self.eval_env_list = []
        self.data_specs_list = []
        self.traj_data_specs_list = []
        
        if self.env_type == 'adroit':
            # reward rescale can either be added in the env or in the agent code when reward is used
            for env_name in env_names:
                train_env = AdroitEnv(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                        num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                        device=self.device, reward_rescale=self.cfg.reward_rescale)
                eval_env = AdroitEnv(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                        num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                        device=self.device, reward_rescale=self.cfg.reward_rescale)

                data_specs = (train_env.observation_spec(),
                        train_env.observation_sensor_spec(),
                        train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'),
                        specs.Array((1,), np.int8, 'n_goal_achieved'),
                        specs.Array((1,), np.float32, 'time_limit_reached'),
                        )
                traj_data_specs = (specs.BoundedArray(shape=(self.cfg.seq_len, train_env.act_dim), dtype='float32', name='action_seq', minimum=-1.0, maximum=1.0),
                                specs.Array((1,), np.float32, name='action_label'),
                        )
                self.train_env_list.append(train_env)
                self.eval_env_list.append(eval_env)
                self.data_specs_list.append(data_specs)
                self.traj_data_specs_list.append(traj_data_specs) 
        else:
            for env_name in env_names:
                train_env = dmc.make(env_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed)
                eval_env = dmc.make(env_name, self.cfg.frame_stack,
                                        self.cfg.action_repeat, self.cfg.seed)
                
                data_specs = (train_env.observation_spec(),
                    train_env.action_spec(),
                    specs.Array((1,), np.float32, 'reward'),
                    specs.Array((1,), np.float32, 'discount'))
                traj_data_specs = (specs.BoundedArray(shape=(self.cfg.seq_len, train_env.action_spec().shape[0]), dtype='float32', name='action_seq', minimum=-1.0, maximum=1.0),
                                specs.Array((1,), np.float32, name='action_label'),
                            )
                self.train_env_list.append(train_env)
                self.eval_env_list.append(eval_env)
                self.data_specs_list.append(data_specs)
                self.traj_data_specs_list.append(traj_data_specs)
        # create replay buffer
        self.replay_storage_list = []
        self.replay_loader_list = []
        self.traj_storage_list = []
        self.traj_loader_list = []
        self._replay_iter_list = []
        self._traj_iter_list = []
        
        for i, env_name in enumerate(env_names):
            replay_storage = ReplayBufferStorage(self.data_specs_list[i], self.work_dir / ('buffer_'+env_name))

            replay_loader = make_replay_loader(
                self.work_dir / ('buffer_'+env_name), self.cfg.replay_buffer_size,
                self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.replay_buffer_fetch_every,
                is_adroit=IS_ADROIT)
            # self._replay_iter = None
            
            traj_storage = TrajBufferStorage(self.traj_data_specs_list[i], self.work_dir / ('buffer_traj_'+env_name))
            self.traj_loader = make_traj_loader(
                self.work_dir / ('buffer_traj_'+env_name), self.cfg.traj_buffer_size,
                self.cfg.traj_batch_size, self.cfg.traj_buffer_num_workers, self.cfg.seq_len, self.traj_buffer_fetch_every,
                self.cfg.save_snapshot, is_adroit=IS_ADROIT)
            # self._traj_iter = None
            
            self.replay_storage_list.append(replay_storage)
            self.replay_loader_list.append(replay_loader)
            self.traj_storage_list.append(traj_storage)
            self.traj_loader_list.append(self.traj_loader)
            self._replay_iter_list.append(None)
            self._traj_iter_list.append(None)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    def set_demo_buffer_nstep(self, nstep):
        self.replay_loader_demo.dataset._nstep = nstep

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    @property
    def env_steps(self):
        return self._env_steps
    
    @property
    def env_episodes(self):
        return self._env_episodes
    
    @property
    def env_frames(self):
        return [self.env_steps[i] * self.cfg.action_repeat for i in range(len(self.env_steps))]

    @property
    def replay_iter(self):
        # if self._replay_iter is None:
        #     self._replay_iter = iter(self.replay_loader)
        # return self._replay_iter
        for i in range(len(self.replay_loader_list)):
            if self._replay_iter_list[i] is None:
                self._replay_iter_list[i] = iter(self.replay_loader_list[i])
        return self._replay_iter_list
    
    @property
    def traj_iter(self):
        # if self._traj_iter is None:
        #     self._traj_iter = iter(self.traj_loader)
        # return self._traj_iter
        for i in range(len(self.traj_loader_list)):
            if self._traj_iter_list[i] is None:
                self._traj_iter_list[i] = iter(self.traj_loader_list[i])
        return self._traj_iter_list

    def eval_dmc(self, i, do_log=False):
        env = self.eval_env_list[i]
        env_step, env_episode, env_total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(env_episode):
            time_step = env.reset()
            self.video_recorder.init(env, enabled=(env_episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                    action = action[self.agent.action_idx[i]:self.agent.action_idx[i+1]]
                time_step = env.step(action)
                self.video_recorder.record(env)
                env_total_reward += time_step.reward
                env_step += 1

            env_episode += 1
            self.video_recorder.save(f'{self.env_names[i]}_{self.env_frames[i]}.mp4')
            if do_log:
                with self.loggers[i].log_and_dump_ctx(self.env_frames[i], ty='eval') as log:
                    # log('episode_reward_'+str(i), env_total_reward / env_episode)
                    # log('episode_length_'+str(i), env_step * self.cfg.action_repeat / env_episode)
                    # log('episode_'+str(i), self.env_episodes[i])
                    # log('step_'+str(i), self.env_steps[i])
                    log('episode_reward', env_total_reward / env_episode)
                    log('episode_length', env_step * self.cfg.action_repeat / env_episode)
                    log('episode', self.env_episodes[i])
                    log('step', self.env_steps[i])
                
                
        # step, episode, total_reward = 0, 0, 0
        # for i, env in enumerate(self.eval_env_list):
        #     env_step, env_episode, env_total_reward = 0, 0, 0
        #     eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        #     while eval_until_episode(env_episode):
        #         time_step = env.reset()
        #         self.video_recorder.init(env, enabled=(env_episode == 0))
        #         while not time_step.last():
        #             with torch.no_grad(), utils.eval_mode(self.agent):
        #                 action = self.agent.act(time_step.observation,
        #                                         self.global_step,
        #                                         eval_mode=True)
        #                 action = action[self.agent.action_idx[i]:self.agent.action_idx[i+1]]
        #             time_step = env.step(action)
        #             self.video_recorder.record(env)
        #             env_total_reward += time_step.reward
        #             total_reward += time_step.reward
        #             env_step += 1

        #         env_episode += 1
        #         episode += 1
        #         self.video_recorder.save(f'{self.env_names[i]}_{self.env_frames[i]}.mp4')
        #     if do_log:
        #         with self.loggers[i].log_and_dump_ctx(self.env_frames[i], ty='eval') as log:
        #             # log('episode_reward_'+str(i), env_total_reward / env_episode)
        #             # log('episode_length_'+str(i), env_step * self.cfg.action_repeat / env_episode)
        #             # log('episode_'+str(i), self.env_episodes[i])
        #             # log('step_'+str(i), self.env_steps[i])
        #             log('episode_reward', env_total_reward / env_episode)
        #             log('episode_length', env_step * self.cfg.action_repeat / env_episode)
        #             log('episode', self.env_episodes[i])
        #             log('step', self.env_steps[i])
        # if do_log:
        #     with self.loggers[-1].log_and_dump_ctx(self.global_frame, ty='eval') as log:
        #         log('episode_reward', total_reward / episode)
        #         log('episode_length', step * self.cfg.action_repeat / episode)
        #         log('episode', self.global_episode)
        #         log('step', self.global_step)
                    
        return env_total_reward / env_episode, 0

    def eval_adroit(self, i, force_number_episodes=None, do_log=True):
        if ENV_TYPE != 'adroit':
            return self.eval_dmc(i, do_log)

        step, episode, total_reward = 0, 0, 0
        n_eval_episode = force_number_episodes if force_number_episodes is not None else self.cfg.num_eval_episodes
        eval_until_episode = utils.Until(n_eval_episode)
        total_success = 0.0
        while eval_until_episode(episode):
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    observation = time_step.observation
                    action = self.agent.act(observation,
                                            self.global_step,
                                            eval_mode=True,
                                            obs_sensor=time_step.observation_sensor)
                time_step = self.eval_env.step(action)
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            # here check if success for Adroit tasks. The threshold values come from the mj_envs code
            # e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
            # can also use the evaluate_success function from Adroit envs, but can be more complicated
            if self.cfg.task_name == 'pen-v0':
                threshold = 20
            else:
                threshold = 25
            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        success_rate_standard = total_success / n_eval_episode
        episode_reward_standard = total_reward / episode
        episode_length_standard = step * self.cfg.action_repeat / episode

        if do_log:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', episode_reward_standard)
                log('success_rate', success_rate_standard)
                log('episode_length', episode_length_standard)
                log('episode', self.global_episode)
                log('step', self.global_step)

        return episode_reward_standard, success_rate_standard

    def get_data_folder_path(self):
        amlt_data_path = os.getenv('AMLT_DATA_DIR', '')
        if amlt_data_path != '':  # if on cluster
            return amlt_data_path
        else: # if not on cluster, return the data folder path specified in cfg
            return self.cfg.local_data_dir

    def get_demo_path(self, env_name):
        # given a environment name (with the -v0 part), return the path to its demo file
        data_folder_path = self.get_data_folder_path()
        demo_folder_path = os.path.join(data_folder_path, "demonstrations")
        demo_path = os.path.join(demo_folder_path, env_name + "_demos.pickle")
        return demo_path

    def load_demo(self, replay_storage_list, traj_storage_list, env_names, verbose=True):
        if self.env_type == 'adroit':
            for i, env_name in enumerate(env_names):
            # will load demo data and put them into a replay storage
                demo_path = self.get_demo_path(env_name)
                if verbose:
                    print("Trying to get demo data from:", demo_path)

                # get the raw state demo data, a list of length 25
                demo_data = pickle.load(open(demo_path, 'rb'))
                if self.cfg.num_demo >= 0:
                    demo_data = demo_data[:self.cfg.num_demo]

                """
                the adroit demo data is in raw state, so we need to convert them into image data
                we init an env and run episodes with the stored actions in the demo data
                then put the image and sensor data into the replay buffer, also need to clip actions here 
                this part is basically following the RRL code
                """
                demo_env = AdroitEnv(env_name, test_image=False, num_repeats=1, num_frames=self.cfg.frame_stack,
                        env_feature_type=self.env_feature_type, device=self.device, reward_rescale=self.cfg.reward_rescale)
                demo_env.reset()

                total_data_count = 0
                for i_path in range(len(demo_data)):
                    path = demo_data[i_path]
                    demo_env.reset()
                    demo_env.set_env_state(path['init_state_dict'])
                    time_step = demo_env.get_current_obs_without_reset()
                    replay_storage_list[i].add(time_step)
                    time_step_traj = dict()
                    time_step_traj['action_seq'] = path['actions'][-self.cfg.seq_len:].astype(np.float32)
                    time_step_traj['action_label'] = 1.0
                    traj_storage_list[i].add(time_step_traj)

                    ep_reward = 0
                    ep_n_goal = 0
                    for i_act in range(len(path['actions'])):
                        total_data_count += 1
                        action = path['actions'][i_act]
                        action = action.astype(np.float32)
                        # when action is put into the environment, they will be clipped.
                        action[action > 1] = 1
                        action[action < -1] = -1

                        # when they collect the demo data, they actually did not use a timelimit...
                        if i_act == len(path['actions']) - 1:
                            force_step_type = 'last'
                        else:
                            force_step_type = 'mid'

                        time_step = demo_env.step(action, force_step_type=force_step_type)
                        replay_storage_list[i].add(time_step)

                        reward = time_step.reward
                        ep_reward += reward

                        goal_achieved = time_step.n_goal_achieved
                        ep_n_goal += goal_achieved
                    if verbose:
                        print('demo trajectory %d, len: %d, return: %.2f, goal achieved steps: %d' %
                            (i_path, len(path['actions']), ep_reward, ep_n_goal))
                for i in range(self.cfg.traj_buffer_size - self.cfg.num_demo):
                    time_step_traj = dict()
                    time_step_traj['action_seq'] = np.random.uniform(-1, 1, (self.cfg.seq_len, self.train_env.act_dim)).astype(np.float32)
                    time_step_traj['action_label'] = 0
                    traj_storage_list[i].add(time_step_traj)
        else:
            for i, env_name in enumerate(env_names):
                total_data_count = 0
                data_folder_path = self.get_data_folder_path()
                pt_path = data_folder_path + '/ckpts/' + env_name + '/' + str(self.cfg.seed) + '/checkpoint-1000000.pt'
                agent = torch.load(pt_path)
                
                for i_demo in range(self.cfg.num_demo):
                    demo_env = dmc.make(env_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed + i_demo)
                    action_shape = demo_env.action_spec().shape[0]
                    total_reward = 0
                    step = 0
                    actions = np.zeros((int(1000/self.cfg.action_repeat), action_shape), dtype=np.float64)
                    
                    time_step = demo_env.reset()
                    replay_storage_list[i].add(time_step)
                    while not time_step.last():
                        with torch.no_grad(), utils.eval_mode(agent):
                            # observations[step] = time_step.observation
                            action = agent.act(time_step.observation,
                                                    5000,
                                                    eval_mode=True)
                            actions[step] = action
                        time_step = demo_env.step(action)
                        replay_storage_list[i].add(time_step)
                        total_reward += time_step.reward
                        step += 1
                        total_data_count += 1
                    time_step_traj = dict()
                    time_step_traj['action_seq'] = actions[-self.cfg.seq_len:].astype(np.float32)
                    time_step_traj['action_label'] = 1.0
                    traj_storage_list[i].add(time_step_traj)
                    if verbose:
                        print('demo trajectory %d, len: %d, return: %.2f' %
                            (i_demo, step, total_reward))
                for _ in range(self.cfg.traj_buffer_size - self.cfg.num_demo):
                    time_step_traj = dict()
                    time_step_traj['action_seq'] = np.random.uniform(-1, 1, (self.cfg.seq_len, demo_env.action_spec().shape[0])).astype(np.float32)
                    time_step_traj['action_label'] = 0
                    traj_storage_list[i].add(time_step_traj)        
            
            
        if verbose:
            print("Demo data load finished, total data count:", total_data_count)

    def get_pretrained_model_path(self, visual_model_name):
        # given a stage1 model name, return the path to the pretrained model
        data_folder_path = self.get_data_folder_path()
        model_folder_path = os.path.join(data_folder_path, "trained_models")
        model_path = os.path.join(model_folder_path, visual_model_name + '_checkpoint.pth.tar')
        return model_path

    def train(self):
        train_start_time = time.time()
        print("\n=== Training started! ===")
        """=================================== LOAD PRETRAINED MODEL ==================================="""
        # if self.cfg.stage1_use_pretrain:
        #     self.agent.load_pretrained_encoder(self.get_pretrained_model_path(self.cfg.visual_model_name))
        # self.agent.switch_to_RL_stages()

        """========================================= LOAD DATA ========================================="""
        if self.cfg.load_demo:
            self.load_demo(self.replay_storage_list, self.traj_storage_list, self.cfg.task_names)
        print("Model and data loading finished in %.2f hours." % ((time.time()-train_start_time) / 3600))
        
        """========================================== STAGE 1 =========================================="""
        print("\n=== Stage 1 started ===")
        stage1_start_time = time.time()
        stage1_n_update = self.cfg.stage1_n_update
        if stage1_n_update > 0:
            for i_stage1 in range(stage1_n_update):
                metrics = self.agent.update_representation(self.replay_iter, self.traj_iter, i_stage1, use_sensor=IS_ADROIT)
                self.loggers[-1].log_metrics(metrics, i_stage1, ty='train')
                if i_stage1 % self.cfg.stage1_eval_every_frames == 0:
                    print('Stage 1 step %d, reconstruction loss: %.3f, inverse dynamics loss: %.3f, forward dynamics loss: %.3f' %
                          (i_stage1, metrics['loss_rec'],  metrics['loss_inv'], metrics['loss_fwd']))
                if self.cfg.show_computation_time_est and i_stage1 > 0 and i_stage1 % self.cfg.show_time_est_interval == 0:
                    print_stage1_time_est(time.time()-stage1_start_time, i_stage1+1, stage1_n_update)

        """========================================== STAGE 2 =========================================="""
        print("\n=== Stage 2 started ===")
        stage2_start_time = time.time()
        stage2_n_update = self.cfg.stage2_n_update
        if stage2_n_update > 0:
            for i_stage2 in range(stage2_n_update):
                metrics = self.agent.update(self.replay_iter, i_stage2, stage='BC', use_sensor=IS_ADROIT)
                if i_stage2 % self.cfg.stage2_eval_every_frames == 0:
                    average_score, succ_rate = 0, 0
                    for i, env in enumerate(self.eval_env_list):
                        score, rate = self.eval_adroit(i, force_number_episodes=self.cfg.stage2_num_eval_episodes,
                                                                do_log=False)
                        average_score += score
                        succ_rate += rate
                    average_score /= len(self.eval_env_list)
                    succ_rate /= len(self.eval_env_list)
                    if ENV_TYPE == 'adroit':
                        print('Stage 2 step %d, Q loss: %.2f, score: %.2f, succ rate: %.2f' %
                             (i_stage2, metrics['critic_loss'], average_score, succ_rate))
                    else:
                        print('Stage 2 step %d, Q loss: %.2f, score: %.2f' %
                             (i_stage2, metrics['critic_loss'], average_score))
                if self.cfg.show_computation_time_est and i_stage2 > 0 and i_stage2 % self.cfg.show_time_est_interval == 0:
                    print_stage2_time_est(time.time()-stage2_start_time, i_stage2+1, stage2_n_update)
        print("Stage 2 finished in %.2f hours." % ((time.time()-stage2_start_time) / 3600))

        """========================================== STAGE 3 =========================================="""
        print("\n=== Stage 3 started ===")
        stage3_start_time = time.time()
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        
        for i, env in enumerate(self.train_env_list):
            episode_step, episode_reward = 0, 0
            time_step = env.reset()
            self.replay_storage_list[i].add(time_step)
            metrics = None

            episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]
            self.timer.reset()
            while train_until_step(self.env_steps[i]):
                # if 1000 steps passed, do some logging
                if self.env_steps[i] % 1000 == 0 and metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame_since_log = episode_step_since_log * self.cfg.action_repeat
                    with self.loggers[i].log_and_dump_ctx(self.env_frames[i], ty='train') as log:
                            log('episode_reward_'+str(i), np.mean(episode_reward_list))
                            log('episode_length_'+str(i), np.mean(episode_frame_list))
                            log('episode', self.env_episodes[i])
                            log('step'+self.env_names[i], self.env_steps[i])
                    episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]
                if self.cfg.show_computation_time_est and self.env_steps[i] > 0 and self.env_steps[i] % self.cfg.show_time_est_interval == 0:
                    print_stage3_time_est(time.time() - stage3_start_time, self.env_frames[i] + 1, self.cfg.num_train_frames)

                # if reached end of episode
                if time_step.last():
                    self._global_episode += 1
                    self._env_episodes[i] += 1
                    
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        episode_step_since_log += episode_step
                        episode_reward_list.append(episode_reward)
                        episode_frame = episode_step * self.cfg.action_repeat
                        episode_frame_list.append(episode_frame)

                    # reset env
                    time_step = env.reset()
                    self.replay_storage_list[i].add(time_step)
                    
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    episode_step, episode_reward = 0, 0

                # try to evaluate
                if eval_every_step(self.env_steps[i]):
                    # self.loggers[i].log('eval_total_time', self.timer.total_time(), self.global_frame)
                    self.eval_adroit(i)

                # sample action
                if IS_ADROIT:
                    obs_sensor = time_step.observation_sensor
                else:
                    obs_sensor = None
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False,
                                        obs_sensor=obs_sensor)
                    action = action[self.agent.action_idx[i]:self.agent.action_idx[i+1]]

                # update the agent
                if not seed_until_step(self.global_step):
                    metrics = self.agent.update(self.replay_iter, self.global_step, stage='DAPG', use_sensor=IS_ADROIT)
                    self.loggers[i].log_metrics(metrics, self.env_frames[i], ty='train')

                # take env step
                time_step = env.step(action)
                episode_reward += time_step.reward
                self.replay_storage_list[i].add(time_step)
                
                episode_step += 1
                self._global_step += 1
                self._env_steps[i] += 1
                """here we move the experiment files to azure blob"""
                if (self._env_steps[i]==1) or (self._env_steps[i] == 10000) or (self._env_steps[i] % 100000 == 0):
                    try:
                        self.copy_to_azure()
                    except Exception as e:
                        print(e)

                """here save model for later"""
            if self.cfg.save_models:
                if self.global_frame in (2, 100000, 500000, 1000000, 2000000, 4000000):
                    self.save_snapshot(suffix=str(self.global_frame))

        try:
            self.copy_to_azure()
        except Exception as e:
            print(e)
        print("Stage 3 finished in %.2f hours." % ((time.time()-stage3_start_time) / 3600))
        print("All stages finished in %.2f hrs. Work dir:" % ((time.time()-train_start_time)/3600))
        print(self.work_dir)

    def save_snapshot(self, suffix=None):
        if suffix is None:
            save_name = 'snapshot.pt'
        else:
            save_name = 'snapshot' + suffix + '.pt'
        snapshot = self.work_dir / save_name
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        print("snapshot saved to:", str(snapshot))

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def copy_to_azure(self):
        amlt_path = os.getenv('AMLT_OUTPUT_DIR', '')
        if amlt_path != '': # if on cluster
            container_log_path = self.work_dir
            amlt_path_to = os.path.join(amlt_path, self.direct_folder_name)
            copy_tree(str(container_log_path), amlt_path_to, update=1)
            # copytree(str(container_log_path), amlt_path_to, dirs_exist_ok=True, ignore=ignore_patterns('*.npy'))
            print("Data copied to:", amlt_path_to)
        # else: # if at local
        #     container_log_path = self.work_dir
        #     amlt_path_to = '/vrl3data/logs'
        #     copy_tree(str(container_log_path), amlt_path_to, update=1)

@hydra.main(config_path='cfgs', config_name='config_tar_xb_v2_'+ENV_TYPE)
def main(cfg):
    # TODO potentially check the task name and decide which libs to load here? 
    W = Workspace
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()