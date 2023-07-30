# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]

        # transpose: 84 x 84 x 3 -> 3 x 84 x 84
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ReverseActionWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        return self._env.step(action[::-1])

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class ExpandActionWrapper(dm_env.Environment):
    def __init__(self, env, id, action_idx, action_min, action_max):
        self._env = env
        self.id = id
        self.action_idx = action_idx
        self.action_min = action_min
        self.action_max = action_max

    def step(self, action):
        action = action[self.action_idx[self.id]:self.action_idx[self.id+1]]
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        spec = self._env.action_spec()
        action_shape = self.action_idx[-1]
        return specs.BoundedArray(shape=(action_shape,), dtype=spec.dtype, minimum=self.action_min, maximum=self.action_max, name=spec.name)

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name) 
    
def make(name, frame_stack, action_repeat, seed, rescale=1.0, reverse_action=False):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    if reverse_action:
        env = ReverseActionWrapper(env)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=env.action_spec().minimum * rescale, maximum=env.action_spec().maximum * rescale)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)

    return env

def make_env(names, type, frame_stack, action_repeat, seed):
    action_specs = []
    env_list = []
    for i, name in enumerate(names):
        domain, task, embodiment = name.split('_')
        domain = dict(cup='ball_in_cup').get(domain, domain)
        if type == 'resize':
            domain = domain + embodiment
            rescale = 1.0
        else: 
            rescale = float(embodiment)
        env = suite.load(domain, task, task_kwargs={'random': seed}, visualize_reward=False)
        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = action_scale.Wrapper(env, minimum=env.action_spec().minimum * rescale, maximum=env.action_spec().maximum * rescale)
        action_specs.append(env.action_spec())
        env_list.append(env)
    
    action_idx = np.cumsum([0] + [spec.shape[0] for spec in action_specs]).tolist()
    action_min = np.concatenate([spec.minimum for spec in action_specs])
    action_max = np.concatenate([spec.maximum for spec in action_specs])

    for i, name in enumerate(names):
        env = ExpandActionWrapper(env_list[i], i, action_idx, action_min, action_max)
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        env = FrameStackWrapper(env, frame_stack, 'pixels')
        env = ExtendedTimeStepWrapper(env)
        env_list[i] = env
        
    return env_list, action_idx
        
def get_current_obs_without_reset(env):
    obs_pixels = env.physics.render(height=84, width=84)
    obs_pixels = obs_pixels.transpose(2, 0, 1).copy()
    obs_pixels = np.concatenate([obs_pixels] * env._num_frames, axis=0)
    action_spec = env.action_spec()
    action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    time_step = ExtendedTimeStep(observation=obs_pixels,
                                step_type=dm_env.StepType.FIRST,
                                action=action,
                                reward=0.0,
                                discount=1.0)
    return time_step
 