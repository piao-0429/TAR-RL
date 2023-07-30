# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, psutil
# make sure mujoco and nvidia will be found
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
                                ':/home/hp/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/home/hp/.mujoco/mujoco210/'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from replay_buffer import ReplayBufferStorage, CrossEmbodimentReplayBufferStorage, make_replay_loader, make_cross_embodiment_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import joblib
import pickle
import time

import envs.dmc.dmc as dmc

torch.backends.cudnn.benchmark = True

def get_ram_info():
    # get info on RAM usage
    d = dict(psutil.virtual_memory()._asdict())
    for key in d:
        if key != "percent": # convert to GB
            d[key] = int(d[key] / 1024**3 * 100)/ 100
    return d

def make_agent(obs_spec, action_spec, action_idx, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_idx = action_idx
    cfg.action_shape = action_spec.shape
    cfg.num_embodiments = len(action_idx)-1
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

        self.agent = make_agent(self.train_env_list[0].observation_spec(),
                                self.train_env_list[0].action_spec(),
                                self.action_idx,
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
        self.cfg.num_demo = 1
        self.cfg.eval_every_frames = 3000
        self.cfg.agent.hidden_dim = 8
        self.cfg.agent.num_expl_steps = 500
        self.cfg.stage2_eval_every_frames = 50
        self.traj_buffer_fetch_every = 2
        self.cfg.stage1_n_update = 10
        self.cfg.stage1_eval_every_frames = 5
        self.cfg.stage2_num_eval_episodes = 1
        self.cfg.num_eval_episodes = 1
        
        
        self.cfg.use_wb = False
        self.cfg.use_tb = False

    def setup(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        if self.cfg.save_models:
            assert self.cfg.action_repeat % 2 == 0

        # create logger
        self.logger = Logger(self.cfg.project, self.cfg.experiment+'-'+str(self.cfg.seed), self.work_dir, use_wb=self.cfg.use_wb, use_tb=self.cfg.use_tb)
        embodiment_type = self.cfg.embodiment_type
        env_names = self.cfg.task_names
        self.env_names = env_names
        self.num_embodiments = len(env_names)
        
        if self.cfg.agent.encoder_lr_scale == 'auto':
            self.cfg.agent.encoder_lr_scale = 1

        self.env_feature_type = self.cfg.env_feature_type
        
        self.train_env_list, self.action_idx = dmc.make_env(env_names, embodiment_type, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
        self.eval_env_list, _ = dmc.make_env(env_names, embodiment_type, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
        
        data_specs = (self.train_env_list[0].observation_spec(),
                      self.train_env_list[0].action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((1,), np.int32, 'embodiment_id')
                    )

        # create replay buffer
        self.replay_storage = CrossEmbodimentReplayBufferStorage(data_specs, self.work_dir / 'buffer', self.num_embodiments)

        self.replay_loader = make_cross_embodiment_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.replay_buffer_fetch_every,
            use_sensor=self.cfg.use_sensor)
        self._replay_iter = None

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
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval_dmc(self, force_number_episodes=None, do_log=False):
        num_episodes = force_number_episodes if force_number_episodes is not None else self.cfg.num_eval_episodes
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(num_episodes)
        for i, env in enumerate(self.eval_env_list):
            env_episode = 0
            while eval_until_episode(env_episode):
                time_step = env.reset()
                self.video_recorder.init(env, enabled=(episode == 0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    time_step = env.step(action)
                    self.video_recorder.record(env)
                    total_reward += time_step.reward
                    step += 1
                env_episode += 1
                episode += 1
                self.video_recorder.save(f'{self.env_names[i]}_{self.global_frame}.mp4')
        if do_log:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
            
        return total_reward / episode

    def get_data_folder_path(self):
        amlt_data_path = os.getenv('AMLT_DATA_DIR', '')
        if amlt_data_path != '':  # if on cluster
            return amlt_data_path
        else: # if not on cluster, return the data folder path specified in cfg
            return self.cfg.local_data_dir

    def load_demo(self, replay_storage, env_names, verbose=True):
        total_data_count = 0
        data_folder_path = self.get_data_folder_path()
        for i_demo in range(self.cfg.num_demo):
            demo_env_list, action_idx = dmc.make_env(env_names, self.cfg.embodiment_type, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed + i_demo)
            action_spec = demo_env_list[0].action_spec()
            for i, env_name in enumerate(env_names):
                if self.cfg.embodiment_type == 'rescale':
                    agent_name, rescale = env_name[:-2], float(env_name[-1])
                else:
                    rescale = 1.0
                pt_path = data_folder_path + '/ckpts/' + agent_name + '/' + str(self.cfg.seed) + '/checkpoint-1000000.pt'
                agent = torch.load(pt_path)
                total_reward = 0
                step = 0
                time_step = demo_env_list[i].reset()                
                replay_storage.add(time_step, i)
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(agent):
                        action = agent.act(time_step.observation, 0, eval_mode=True) * rescale
                        expand_action = np.zeros(action_spec.shape, dtype=np.float32)
                        expand_action[action_idx[i]:action_idx[i+1]] = action
    
                        # expand the action to match the action spec
                    time_step = demo_env_list[i].step(expand_action)
                    replay_storage.add(time_step, i)
                    total_reward += time_step.reward
                    step += 1
                    total_data_count += 1
                if verbose:
                    print('demo trajectory %d for embodiment %s, len: %d, return: %.2f' %
                        (i_demo, env_name, step, total_reward))
                
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
        """========================================= LOAD DATA ========================================="""
        if self.cfg.load_demo:
            self.load_demo(self.replay_storage, self.cfg.task_names)
        print("Model and data loading finished in %.2f hours." % ((time.time()-train_start_time) / 3600))
        
        """========================================== STAGE 1 =========================================="""
        print("\n=== Stage 1 started ===")
        stage1_start_time = time.time()
        stage1_n_update = self.cfg.stage1_n_update
        if stage1_n_update > 0:
            for i_stage1 in range(stage1_n_update):
                metrics = self.agent.update_representation(self.replay_iter, i_stage1, use_sensor=self.cfg.use_sensor)
                self.logger.log_metrics(metrics, i_stage1, ty='train')
                if i_stage1 % self.cfg.stage1_eval_every_frames == 0:
                    print('Stage 1 step %d, reconstruction loss: %.3f, inverse dynamics loss: %.3f, forward dynamics loss: %.3f' %
                          (i_stage1, metrics['loss_rec_pretrain'],  metrics['loss_inv_pretrain'], metrics['loss_fwd_pretrain']))
                if self.cfg.show_computation_time_est and i_stage1 > 0 and i_stage1 % self.cfg.show_time_est_interval == 0:
                    print_stage1_time_est(time.time()-stage1_start_time, i_stage1+1, stage1_n_update)

        """========================================== STAGE 2 =========================================="""
        print("\n=== Stage 2 started ===")
        stage2_start_time = time.time()
        stage2_n_update = self.cfg.stage2_n_update
        if stage2_n_update > 0:
            for i_stage2 in range(stage2_n_update):
                metrics = self.agent.update(self.replay_iter, i_stage2, stage='BC', use_sensor=self.cfg.use_sensor)
                if i_stage2 % self.cfg.stage2_eval_every_frames == 0:
                    average_score = self.eval_dmc(force_number_episodes=self.cfg.stage2_num_eval_episodes,
                                                                do_log=False)
                    print('Stage 2 step %d, Q(s,a): %.2f, Q loss: %.2f, score: %.2f' %
                          (i_stage2, metrics['critic_q1'],  metrics['critic_loss'], average_score))
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

        # episode_step, episode_reward = 0, 0
        time_steps = []
        episode_steps, episode_rewards = [], []
        for i, train_env in enumerate(self.train_env_list):
            time_step = train_env.reset()
            time_steps.append(time_step)
            self.replay_storage.add(time_step, i)
            episode_steps.append(0)
            episode_rewards.append(0)
            
        metrics = None

        episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]
        self.timer.reset()
        while train_until_step(self.global_step):
            # if 1000 steps passed, do some logging
            # TODO
            if self.global_step % 1000 == 0 and metrics is not None:
                elapsed_time, total_time = self.timer.reset()
                episode_frame_since_log = episode_step_since_log * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame_since_log / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', np.mean(episode_reward_list))
                        log('episode_length', np.mean(episode_frame_list))
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]
            if self.cfg.show_computation_time_est and self.global_step > 0 and self.global_step % self.cfg.show_time_est_interval == 0:
                print_stage3_time_est(time.time() - stage3_start_time, self.global_frame + 1, self.cfg.num_train_frames)

            # if reached end of episode
            for i, time_step in enumerate(time_steps):
                if time_step.last():
                    self._global_episode += 1

                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        episode_step_since_log += episode_steps[i]
                        episode_reward_list.append(episode_rewards[i])
                        episode_frame = episode_steps[i] * self.cfg.action_repeat
                        episode_frame_list.append(episode_frame)

                    # reset env
                    time_steps[i] = self.train_env_list[i].reset()
                    self.replay_storage.add(time_steps[i], i)

                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    episode_steps[i], episode_rewards[i] = 0, 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval_dmc()

            # sample action
            if self.cfg.use_sensor:
                obs_sensor = time_step.observation_sensor
            else:
                obs_sensor = None
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                    self.global_step,
                                    eval_mode=False,
                                    obs_sensor=obs_sensor)

            # update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step, stage='DAPG', use_sensor=self.cfg.use_sensor)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            for i, train_env in enumerate(self.train_env_list):
                time_steps[i] = train_env.step(action)
                episode_rewards[i] += time_steps[i].reward
                self.replay_storage.add(time_steps[i], i)
                episode_steps[i] += 1
                
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
            # copytree(str(container_log_path), amlt_path_to, dirs_exist_ok=True, ignore=ignore_patterns('*.npy'))
            print("Data copied to:", amlt_path_to)
        # else: # if at local
        #     container_log_path = self.work_dir
        #     amlt_path_to = '/vrl3data/logs'
        #     copy_tree(str(container_log_path), amlt_path_to, update=1)

@hydra.main(config_path='cfgs', config_name='config_xar_dmc')
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