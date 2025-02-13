# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, psutil
# make sure mujoco and nvidia will be found
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
                                ':/home/hp/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/home/hp/.mujoco/mujoco210/'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
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

        self.agent = make_agent(self.train_env_1.observation_spec(),
                                self.train_env_1.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

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
        
        self.cfg.traj_buffer_size = 4
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
        self.logger = Logger(self.cfg.project, self.cfg.experiment+'-'+str(self.cfg.seed), self.work_dir, use_wb=self.cfg.use_wb, use_tb=self.cfg.use_tb)
        env_name = self.cfg.task_name
        env_type = 'adroit' if env_name in ('hammer-v0','door-v0','pen-v0','relocate-v0') else 'dmc'
        # assert env_name in ('hammer-v0','door-v0','pen-v0','relocate-v0',)
        self.env_type = env_type
        if self.cfg.agent.encoder_lr_scale == 'auto':
            if env_name == 'relocate-v0':
                self.cfg.agent.encoder_lr_scale = 0.01
            else:
                self.cfg.agent.encoder_lr_scale = 1

        self.env_feature_type = self.cfg.env_feature_type
        if env_type == 'adroit':
            # reward rescale can either be added in the env or in the agent code when reward is used
            self.train_env = AdroitEnv(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                    num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                       device=self.device, reward_rescale=self.cfg.reward_rescale)
            self.eval_env = AdroitEnv(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                    num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                      device=self.device, reward_rescale=self.cfg.reward_rescale)

            data_specs = (self.train_env.observation_spec(),
                      self.train_env.observation_sensor_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((1,), np.int8, 'n_goal_achieved'),
                      specs.Array((1,), np.float32, 'time_limit_reached'),
                      )
            traj_data_specs = (specs.BoundedArray(shape=(self.cfg.seq_len, self.train_env.act_dim), dtype='float32', name='action_seq', minimum=-1.0, maximum=1.0),
                               specs.Array((1,), np.float32, name='action_label'),
                      ) 
        else:
            self.train_env_1 = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
            self.train_env_2 = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, reverse_action=True)
            self.eval_env_1 = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed)
            self.eval_env_2 = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed, reverse_action=True)
            
            data_specs = (self.train_env_1.observation_spec(),
                self.train_env_1.action_spec(),
                specs.Array((1,), np.float32, 'reward'),
                specs.Array((1,), np.float32, 'discount'))
            traj_data_specs = (specs.BoundedArray(shape=(self.cfg.seq_len, self.train_env_1.action_spec().shape[0]), dtype='float32', name='action_seq', minimum=-1.0, maximum=1.0),
                            specs.Array((1,), np.float32, name='action_label'),
                        )

        # create replay buffer
        self.replay_storage_1 = ReplayBufferStorage(data_specs, self.work_dir / 'buffer_1')
        self.replay_storage_2 = ReplayBufferStorage(data_specs, self.work_dir / 'buffer_2')

        self.replay_loader_1 = make_replay_loader(
            self.work_dir / 'buffer_1', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.replay_buffer_fetch_every,
            is_adroit=IS_ADROIT)
        self._replay_iter_1 = None
        
        self.replay_loader_2 = make_replay_loader(
            self.work_dir / 'buffer_2', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.replay_buffer_fetch_every,
            is_adroit=IS_ADROIT)
        self._replay_iter_2 = None
        
        self.traj_storage_1 = TrajBufferStorage(traj_data_specs, self.work_dir / 'buffer_traj_1')
        self.traj_loader_1 = make_traj_loader(
            self.work_dir / 'buffer_traj_1', self.cfg.traj_buffer_size,
            self.cfg.traj_batch_size, self.cfg.traj_buffer_num_workers, self.cfg.seq_len, self.traj_buffer_fetch_every,
            self.cfg.save_snapshot, is_adroit=IS_ADROIT)
        self._traj_iter_1 = None
        
        self.traj_storage_2 = TrajBufferStorage(traj_data_specs, self.work_dir / 'buffer_traj_2')
        self.traj_loader_2 = make_traj_loader(
            self.work_dir / 'buffer_traj_2', self.cfg.traj_buffer_size,
            self.cfg.traj_batch_size, self.cfg.traj_buffer_num_workers, self.cfg.seq_len, self.traj_buffer_fetch_every,
            self.cfg.save_snapshot, is_adroit=IS_ADROIT)
        self._traj_iter_2 = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

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
    def replay_iter(self):
        if self._replay_iter_1 is None:
            self._replay_iter_1 = iter(self.replay_loader_1)
        if self._replay_iter_2 is None:
            self._replay_iter_2 = iter(self.replay_loader_2)
        return [self._replay_iter_1, self._replay_iter_2]
    
    @property
    def traj_iter(self):
        if self._traj_iter_1 is None:
            self._traj_iter_1 = iter(self.traj_loader_1)
        if self._traj_iter_2 is None:
            self._traj_iter_2 = iter(self.traj_loader_2)
        return [self._traj_iter_1, self._traj_iter_2]

    def eval_dmc(self, do_log=False, reverse_action=False):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        if reverse_action == 0:
            eval_env = self.eval_env_1
        else:
            eval_env = self.eval_env_2
        while eval_until_episode(episode):
            time_step = eval_env.reset()
            self.video_recorder.init(eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True, reverse_action=reverse_action)
                time_step = eval_env.step(action)
                self.video_recorder.record(eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{1-int(reverse_action)}.mp4')
        if do_log:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
            
        return total_reward / episode, 0

    def eval_adroit(self, force_number_episodes=None, do_log=True, reverse_action=False):
        if ENV_TYPE != 'adroit':
            return self.eval_dmc(do_log, reverse_action)

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

    def load_demo(self, replay_storage_1, replay_storage_2, traj_storage_1,  traj_storage_2, env_name, verbose=True):
        if self.env_type == 'adroit':
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
                replay_storage_1.add(time_step)
                time_step_traj = dict()
                time_step_traj['action_seq'] = path['actions'][-self.cfg.seq_len:].astype(np.float32)
                time_step_traj['action_label'] = 1.0
                traj_storage_1.add(time_step_traj)

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
                    replay_storage_1.add(time_step)

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
                traj_storage_1.add(time_step_traj)
        else:
            total_data_count = 0
            data_folder_path = self.get_data_folder_path()
            pt_path = data_folder_path + '/ckpts/' + self.cfg.task_name + '/1/checkpoint-1000000.pt'
            agent = torch.load(pt_path)
            
            for i_demo in range(self.cfg.num_demo):
                if i_demo % 2 == 0:
                    demo_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed + i_demo)
                else:
                    demo_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed + i_demo, reverse_action=True)
                action_shape = demo_env.action_spec().shape[0]
                total_reward = 0
                step = 0
                actions = np.zeros((int(1000/self.cfg.action_repeat), action_shape), dtype=np.float64)
                
                time_step = demo_env.reset()
                if i_demo % 2 == 0:
                    replay_storage_1.add(time_step)
                else:
                    replay_storage_2.add(time_step)
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(agent):
                        # observations[step] = time_step.observation
                        action = agent.act(time_step.observation,
                                                5000,
                                                eval_mode=True)
                        if i_demo % 2 == 1:
                            action = action[::-1]
                        actions[step] = action
                    time_step = demo_env.step(action)
                    if i_demo % 2 == 0:
                        replay_storage_1.add(time_step)
                    else:
                        replay_storage_2.add(time_step)
                    total_reward += time_step.reward
                    step += 1
                    total_data_count += 1
                time_step_traj = dict()
                time_step_traj['action_seq'] = actions[-self.cfg.seq_len:].astype(np.float32)
                time_step_traj['action_label'] = 1.0
                if i_demo % 2 == 0:
                    traj_storage_1.add(time_step_traj)
                else:
                    traj_storage_2.add(time_step_traj)
                if verbose:
                    print('demo trajectory %d, len: %d, return: %.2f, embodiment: %d' %
                        (i_demo, step, total_reward, i_demo % 2))
            for i in range(self.cfg.traj_buffer_size - self.cfg.num_demo):
                time_step_traj = dict()
                time_step_traj['action_seq'] = np.random.uniform(-1, 1, (self.cfg.seq_len, self.train_env_1.action_spec().shape[0])).astype(np.float32)
                time_step_traj['action_label'] = 0
                if i % 2 == 0:
                    traj_storage_1.add(time_step_traj)
                else:
                    traj_storage_2.add(time_step_traj)       
            
            
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
            self.load_demo(self.replay_storage_1, self.replay_storage_2, self.traj_storage_1, self.traj_storage_2, self.cfg.task_name)
        print("Model and data loading finished in %.2f hours." % ((time.time()-train_start_time) / 3600))
        
        """========================================== STAGE 1 =========================================="""
        print("\n=== Stage 1 started ===")
        stage1_start_time = time.time()
        stage1_n_update = self.cfg.stage1_n_update
        if stage1_n_update > 0:
            for i_stage1 in range(stage1_n_update):
                metrics = self.agent.update_representation(self.replay_iter, self.traj_iter, i_stage1, use_sensor=IS_ADROIT)
                self.logger.log_metrics(metrics, i_stage1, ty='train')
                if i_stage1 % self.cfg.stage1_eval_every_frames == 0:
                    print('Stage 1 step %d, reconstruction loss: %.3f, cycle consistency loss: %.3f' %
                          (i_stage1, metrics['loss_rec'],  metrics['loss_cycle_rec']))
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
                    average_score_1, succ_rate_1 = self.eval_adroit(force_number_episodes=self.cfg.stage2_num_eval_episodes,
                                                                do_log=False, reverse_action=False)
                    average_score_2, succ_rate_2 = self.eval_adroit(force_number_episodes=self.cfg.stage2_num_eval_episodes,
                                                                do_log=False, reverse_action=True)
                    print('Stage 2 step %d, score-1: %.2f, succ rate-1: %.2f, score-2: %.2f, succ rate-2: %.2f' %
                          (i_stage2, average_score_1, succ_rate_1, average_score_2, succ_rate_2))
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

        episode_step, episode_reward_1, episode_reward_2 = 0, 0, 0
        time_step_1 = self.train_env_1.reset()
        self.replay_storage_1.add(time_step_1)
        time_step_2 = self.train_env_2.reset()
        self.replay_storage_2.add(time_step_2)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None

        episode_step_since_log, episode_reward_list_1, episode_reward_list_2, episode_frame_list = 0, [0], [0], [0]
        self.timer.reset()
        while train_until_step(self.global_step):
            # if 1000 steps passed, do some logging
            if self.global_step % 1000 == 0 and metrics is not None:
                elapsed_time, total_time = self.timer.reset()
                episode_frame_since_log = episode_step_since_log * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame_since_log / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward_1', np.mean(episode_reward_list_1))
                        log('episode_reward_2', np.mean(episode_reward_list_2))
                        log('episode_length', np.mean(episode_frame_list))
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage_1))
                        log('step', self.global_step)
                episode_step_since_log, episode_reward_list_1, episode_reward_list_2, episode_frame_list = 0, [0], [0], [0]
            if self.cfg.show_computation_time_est and self.global_step > 0 and self.global_step % self.cfg.show_time_est_interval == 0:
                print_stage3_time_est(time.time() - stage3_start_time, self.global_frame + 1, self.cfg.num_train_frames)

            # if reached end of episode
            if time_step_1.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    episode_step_since_log += episode_step
                    episode_reward_list_1.append(episode_reward_1)
                    episode_reward_list_2.append(episode_reward_2)
                    episode_frame = episode_step * self.cfg.action_repeat
                    episode_frame_list.append(episode_frame)

                # reset env
                time_step_1 = self.train_env_1.reset()
                self.replay_storage_1.add(time_step_1)
                time_step_2 = self.train_env_2.reset()
                self.replay_storage_2.add(time_step_2)
                # self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step, episode_reward_1, episode_reward_2 = 0, 0, 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval_adroit(reverse_action=False)
                self.eval_adroit(reverse_action=True)

            # sample action
            if IS_ADROIT:
                obs_sensor = time_step.observation_sensor
            else:
                obs_sensor = None
            with torch.no_grad(), utils.eval_mode(self.agent):
                action_1 = self.agent.act(time_step_1.observation,
                                    self.global_step,
                                    eval_mode=False,
                                    obs_sensor=obs_sensor, reverse_action=False)
                action_2 = self.agent.act(time_step_2.observation,  
                                    self.global_step,
                                    eval_mode=False,
                                    obs_sensor=obs_sensor, reverse_action=True)

            # update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step, stage='DAPG', use_sensor=IS_ADROIT)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step_1 = self.train_env_1.step(action_1)
            episode_reward_1 += time_step_1.reward
            self.replay_storage_1.add(time_step_1)
            time_step_2 = self.train_env_2.step(action_2)
            episode_reward_2 += time_step_2.reward
            self.replay_storage_2.add(time_step_2)
            # self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            """here we move the experiment files to azure blob"""
            if (self.global_step==1) or (self.global_step == 10000) or (self.global_step % 100000 == 0):
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
            print("Data copied to:", amlt_path_to)


@hydra.main(config_path='cfgs', config_name='config_tar_xb_'+ENV_TYPE)
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