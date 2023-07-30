# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class TrajBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_episodes

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            # print(spec.name, spec.shape, spec.dtype, value.shape, value.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        episode = dict()
        for spec in self._data_specs:
            value = self._current_episode[spec.name]
            episode[spec.name] = np.array(value, spec.dtype)
        self._current_episode = defaultdict(list)
        self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class TrajBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, seq_len, fetch_every,
                 save_snapshot, is_adroit=False, return_next_action=False):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self.seq_len = seq_len
        self._episode_fns = []
        self._episodes = dict()
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._is_adroit = is_adroit
        self._return_next_action = return_next_action


    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += 1

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True
    
    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, _ = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + 1 > self._max_size:
                break
            fetched_size += 1
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        actions = episode['action_seq'][0]
        label = episode['action_label'][0]
        return actions, label

    def __iter__(self):
        while True:
            yield self._sample()

class CrossEmbodimentTrajBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, seq_len, fetch_every,
                 save_snapshot, is_adroit=False, return_next_action=False):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self.seq_len = seq_len
        self._episode_fns = []
        self._episodes = dict()
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._is_adroit = is_adroit
        self._return_next_action = return_next_action


    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += 1

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True
    
    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, _ = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + 1 > self._max_size:
                break
            fetched_size += 1
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        actions = episode['action_seq'][0]
        label = episode['action_label'][0]
        embodiment = episode['embodiment'][0]
        return actions, label, embodiment

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_traj_loader(replay_dir, max_size, batch_size, num_workers, seq_len, fetch_every, 
                       save_snapshot, is_adroit=False, return_next_action=False):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = TrajBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            seq_len,
                            fetch_every=fetch_every,
                            save_snapshot=save_snapshot, 
                            is_adroit=is_adroit,
                            return_next_action=return_next_action)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader

def reinit_data_loader(data_loader, batch_size, num_workers):
    # reinit a data loader with a new batch size
    loader = torch.utils.data.DataLoader(data_loader.dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader