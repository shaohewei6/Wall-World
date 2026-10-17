"""Read recorded trajectories without importing historical experiment modules."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import sha256

MOVES = np.asarray([(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1)], dtype=np.int64)
BACK = np.asarray([1,0,3,2,7,6,5,4], dtype=np.int64)
ARRAYS = ('observations','actions','states','collision_flags')


def action_labels(actions, collisions, start, stop):
    frames = np.arange(start, stop+1)
    valid = frames < len(actions)
    action = np.zeros(len(frames), np.int64)
    hit = np.zeros(len(frames), bool)
    action[valid], hit[valid] = actions[frames[valid]], collisions[frames[valid]]
    previous, back = frames-1, np.full(len(frames), -1, np.int64)
    incoming = previous >= 0
    incoming[incoming] = ~collisions[previous[incoming]]
    back[incoming] = BACK[actions[previous[incoming]]]
    return dict(policy_action=action, policy_collision=hit, policy_back=back, policy_valid=valid)


class OfflineEpisodes:
    """LRU of read-only mmap episodes, or NPZ archives as a compatibility path.

    Batches contain only the requested RGB slice. No visual-feature disk cache is
    created. NPZ needs whole-episode decompression; prefer the NPY mmap layout.
    """
    def __init__(self, root, *, image_size, patch_size, grid_size, episode_ids=None, cache_size=4):
        self.root = Path(root).expanduser().resolve()
        directories = {int(p.name[8:]): p for p in self.root.glob('episode_*')
                       if p.is_dir() and p.name[8:].isdigit()}
        archives = {int(p.stem[8:]): p for p in (self.root/'episodes').glob('episode_*.npz')
                    if p.stem[8:].isdigit()}
        if not directories:
            archives.update({int(p.stem[8:]):p for p in self.root.glob('episode_*.npz') if p.stem[8:].isdigit()})
        self.paths = directories or archives
        ids = sorted(self.paths) if episode_ids is None else list(episode_ids)
        if not ids or len(ids) != len(set(ids)) or any(i not in self.paths for i in ids):
            raise ValueError('Episode selection is empty, duplicated or unavailable')
        self.ids = sorted(ids)
        self.paths = {i:self.paths[i] for i in self.ids}
        self.image_size, self.patch_size, self.grid_size = image_size, patch_size, grid_size
        self.cache_size, self.cache = max(1,cache_size), OrderedDict()
        self.inventory = []
        for i,p in self.paths.items():
            if p.is_dir():
                files = [p/(k+'.npy') for k in ARRAYS]
                if not all(f.is_file() for f in files):
                    raise ValueError(f'Incomplete NPY episode: {p}')
                complete = p/'complete.json'
                # RGB bytes are not rescanned; metadata/stat identity is recorded.
                details = {f.name:dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns,
                    sha256=sha256(f) if f.name!='observations.npy' else None) for f in files}
                if complete.is_file():details['complete.json'] = sha256(complete)
            else:
                details = dict(size=p.stat().st_size,sha256=sha256(p))
            self.inventory.append(dict(episode_id=i,path=str(p),files=details))
        self.fingerprint = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        self.transitions = len(self.load(self.ids[0])['actions'])
        if self.transitions > 4096:
            raise ValueError('The current K6 schedule supports at most 4096 transitions per episode')

    def load(self, episode):
        if episode in self.cache:
            self.cache.move_to_end(episode)
            return self.cache[episode]
        p = self.paths[episode]
        if p.is_dir():
            values = {k:np.load(p/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ARRAYS}
        else:
            with np.load(p,allow_pickle=False) as data:values = {k:data[k] for k in ARRAYS}
        rgb,actions,states,hits = (values[k] for k in ARRAYS)
        n = len(actions)
        if n < 1 or (hasattr(self,'transitions') and n != self.transitions):
            raise ValueError('Stateful episode batches require equal positive episode lengths')
        if rgb.shape != (n+1,3,self.image_size,self.image_size) or rgb.dtype != np.uint8:
            raise ValueError(f'Invalid RGB shape/dtype in {p}')
        if actions.shape != (n,) or actions.dtype.kind not in 'ui' or np.any(actions<0) or np.any(actions>=8):
            raise ValueError(f'Invalid recorded actions in {p}')
        if states.shape != (n+1,2) or states.dtype.kind not in 'ui' or hits.shape != (n,) or hits.dtype != bool:
            raise ValueError(f'Invalid state/collision arrays in {p}')
        patch = (states.astype(np.float64)-self.patch_size/2)/self.patch_size
        if not np.allclose(patch,np.rint(patch),atol=1e-7,rtol=0):
            raise ValueError(f'Pixel centres are not on the patch lattice in {p}')
        patch = np.rint(patch).astype(np.int64)
        if np.any(patch<0) or np.any(patch>=self.grid_size):
            raise ValueError(f'Recorded coordinates exceed configured world in {p}')
        delta = patch[1:]-patch[:-1]
        if not np.array_equal(delta,np.where(hits[:,None],0,MOVES[actions])):
            raise ValueError(f'Action/collision/position feedback disagrees in {p}')
        values['positions'] = patch
        self.cache[episode] = values
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return values

    def batch(self, ids, start, length):
        stop = min(start+length,self.transitions)
        if not 0<=start<stop<=self.transitions:raise ValueError('Invalid trajectory slice')
        rows = []
        for i in ids:
            e = self.load(i)
            row = dict(observations=e['observations'][start:stop+1],
                       actions=e['actions'][start:stop].astype(np.int64),
                       relative_positions=e['positions'][start:stop+1]-e['positions'][0],
                       collision_flags=e['collision_flags'][start:stop])
            row.update(action_labels(e['actions'],e['collision_flags'],start,stop))
            rows.append(row)
        return {k:torch.from_numpy(np.stack([row[k] for row in rows])) for k in rows[0]}, stop

    def group(self, seed, epoch, group_index, batch_size):
        order = np.random.default_rng(np.random.SeedSequence([seed,epoch,190])).permutation(self.ids)
        return order[group_index*batch_size:(group_index+1)*batch_size].tolist()
