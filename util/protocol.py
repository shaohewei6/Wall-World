"""Episode-global uniform segment lengths; correction follows the segment's last prior."""
from functools import lru_cache
import math
import numpy as np
import torch


@lru_cache(maxsize=512)
def schedule(seed, epoch, episode, maximum, length=4096):
    if not 1 <= maximum <= 6: raise ValueError('maximum must be 1..6')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, episode]))
    observe = np.zeros(length, dtype=bool)
    age = np.zeros(length, dtype=np.int64)
    remaining = np.zeros(length, dtype=np.int64)
    start = 0
    while start < length:
        span = int(rng.integers(1, maximum+1))
        end = min(start+span, length)
        age[start:end] = np.arange(end-start)
        remaining[start:end] = span-np.arange(end-start)
        if start+span <= length: observe[end-1] = True
        start = end
    for a in (observe, age, remaining): a.setflags(write=False)
    return observe, age, remaining


def mode_batch(ids, starts, steps, *, seed, epoch, maximum, length=4096, all_observed=False):
    modes, ages = [], []
    for ep, start in zip(ids, starts):
        if not 0 <= start < start+steps <= length: raise ValueError('Invalid slice')
        mask, age, _ = schedule(seed, epoch, ep, maximum, length)
        modes.append(np.ones(steps, bool) if all_observed else mask[start:start+steps])
        ages.append(0 if all_observed else int(age[start]))
    return torch.from_numpy(np.stack(modes)), torch.tensor(ages, dtype=torch.long)


def cursor_state(ids, start, **kwargs):
    result = []
    for ep in ids:
        _, age, remaining = schedule(episode=ep, **kwargs)
        result.append({'episode': ep, 'start': start, 'depth_before': int(age[start]),
                       'steps_until_correction': int(remaining[start])})
    return result


def coordinate_variance(completed_steps, anneal_steps=2192):
    progress = min(max(completed_steps/anneal_steps, 0.), 1.)
    return .5 + .25*(1.+math.cos(math.pi*progress))
