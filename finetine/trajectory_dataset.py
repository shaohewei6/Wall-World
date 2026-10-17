"""Unlabelled temporal clips from the large-patch Wall World trajectories."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class _EpisodeObservationCache:
    """Bounded, per-worker cache of decompressed episode observations."""

    def __init__(self, episode_cache_size: int) -> None:
        if episode_cache_size < 0:
            raise ValueError("episode_cache_size must be non-negative")
        self.episode_cache_size = int(episode_cache_size)
        self._episode_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _observations_for(self, episode_index: int) -> np.ndarray:
        cached = self._episode_cache.pop(episode_index, None)
        if cached is not None:
            self._episode_cache[episode_index] = cached
            return cached
        with np.load(self._archive_for(episode_index), allow_pickle=False) as payload:
            observations = payload["observations"]
        if observations.ndim != 4 or observations.shape[1:] != (3, 224, 224):
            raise ValueError(f"Unexpected observation array shape for episode {episode_index}: {observations.shape}")
        if self.episode_cache_size:
            self._episode_cache[episode_index] = observations
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return observations


class LargePatchClipDataset(_EpisodeObservationCache, Dataset[torch.Tensor]):
    """Read fixed-stride video clips without using actions, states, or labels.

    This is deliberately an unlabelled dataset view: V-JEPA pretraining sees
    only temporal RGB observations.  The train/test split stays intact and
    action/collision metadata remains available only for later evaluation.
    """

    def __init__(
        self,
        split_root: str | Path,
        *,
        clip_length: int = 8,
        temporal_stride: int = 1,
        start_stride: int = 1,
        episode_cache_size: int = 1,
    ) -> None:
        _EpisodeObservationCache.__init__(self, episode_cache_size)
        self.split_root = Path(split_root).expanduser().resolve()
        self.clip_length = int(clip_length)
        self.temporal_stride = int(temporal_stride)
        self.start_stride = int(start_stride)
        if self.clip_length < 2 or self.temporal_stride < 1 or self.start_stride < 1:
            raise ValueError("clip_length must be >=2 and strides must be positive")
        listing = self.split_root / "episodes.jsonl"
        if not listing.is_file():
            raise FileNotFoundError(f"Missing episode index: {listing}")
        self.episode_indices: list[int] = []
        with listing.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.episode_indices.append(int(json.loads(line)["episode_index"]))
        if not self.episode_indices:
            raise ValueError(f"No episode records found in {listing}")
        first_archive = self._archive_for(self.episode_indices[0])
        with np.load(first_archive, allow_pickle=False) as payload:
            observations = payload["observations"]
            if observations.ndim != 4 or observations.shape[1:] != (3, 224, 224):
                raise ValueError(f"Unexpected observation array in {first_archive}: {observations.shape}")
            self.frames_per_episode = int(observations.shape[0])
        self.frame_span = (self.clip_length - 1) * self.temporal_stride + 1
        self.starts_per_episode = (self.frames_per_episode - self.frame_span) // self.start_stride + 1
        if self.starts_per_episode < 1:
            raise ValueError("Configured clip exceeds the available trajectory length")

    @property
    def samples_per_episode(self) -> int:
        return self.starts_per_episode

    def _archive_for(self, episode_index: int) -> Path:
        archive = self.split_root / "episodes" / f"episode_{episode_index:04d}.npz"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing episode archive: {archive}")
        return archive

    def __len__(self) -> int:
        return len(self.episode_indices) * self.starts_per_episode

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_slot, start_slot = divmod(index, self.starts_per_episode)
        episode_index = self.episode_indices[episode_slot]
        start = start_slot * self.start_stride
        stop = start + self.frame_span
        frames = self._observations_for(episode_index)[start:stop:self.temporal_stride]
        if frames.shape != (self.clip_length, 3, 224, 224):
            raise RuntimeError(f"Invalid clip shape {frames.shape} for episode={episode_index}, start={start}")
        return torch.from_numpy(frames)


class LargePatchFrameDataset(_EpisodeObservationCache, Dataset[torch.Tensor]):
    """Unlabelled individual observations for image-based DINOv2 SSL."""

    def __init__(
        self,
        split_root: str | Path,
        *,
        frame_stride: int = 1,
        episode_cache_size: int = 1,
    ) -> None:
        _EpisodeObservationCache.__init__(self, episode_cache_size)
        self.split_root = Path(split_root).expanduser().resolve()
        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        listing = self.split_root / "episodes.jsonl"
        if not listing.is_file():
            raise FileNotFoundError(f"Missing episode index: {listing}")
        with listing.open("r", encoding="utf-8") as handle:
            self.episode_indices = [int(json.loads(line)["episode_index"]) for line in handle if line.strip()]
        if not self.episode_indices:
            raise ValueError(f"No episode records found in {listing}")
        with np.load(self._archive_for(self.episode_indices[0]), allow_pickle=False) as payload:
            observations = payload["observations"]
            if observations.ndim != 4 or observations.shape[1:] != (3, 224, 224):
                raise ValueError(f"Unexpected observation array shape: {observations.shape}")
            self.frames_per_episode = int(observations.shape[0])
        self.frames_per_episode_sampled = (self.frames_per_episode - 1) // self.frame_stride + 1

    @property
    def samples_per_episode(self) -> int:
        return self.frames_per_episode_sampled

    def _archive_for(self, episode_index: int) -> Path:
        archive = self.split_root / "episodes" / f"episode_{episode_index:04d}.npz"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing episode archive: {archive}")
        return archive

    def __len__(self) -> int:
        return len(self.episode_indices) * self.frames_per_episode_sampled

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_slot, frame_slot = divmod(index, self.frames_per_episode_sampled)
        frame = self._observations_for(self.episode_indices[episode_slot])[frame_slot * self.frame_stride]
        return torch.from_numpy(frame)


class EpisodeBatchSampler(Sampler[list[int]]):
    """Shuffle batches globally while keeping every batch inside one episode.

    A compressed episode is decoded once per worker cache residency rather than
    once per sampled frame or clip. The order is deterministic for a seed and
    epoch, allowing a resumed run to skip completed batches safely.
    """

    def __init__(
        self,
        dataset: LargePatchClipDataset | LargePatchFrameDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._samples_per_episode = int(dataset.samples_per_episode)
        self._batches_per_episode = (
            self._samples_per_episode // self.batch_size
            if self.drop_last
            else math.ceil(self._samples_per_episode / self.batch_size)
        )
        if self._batches_per_episode < 1:
            raise ValueError("batch_size is larger than the available samples per episode")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset.episode_indices) * self._batches_per_episode

    def __iter__(self) -> Iterator[list[int]]:
        episode_count = len(self.dataset.episode_indices)
        if self.shuffle:
            root_rng = np.random.default_rng(self.seed + self.epoch)
            batch_order = root_rng.permutation(len(self)).astype(np.int64, copy=False)
        else:
            batch_order = np.arange(len(self), dtype=np.int64)
        permutations: dict[int, np.ndarray] = {}
        for flat_batch in batch_order:
            episode_slot, batch_slot = divmod(int(flat_batch), self._batches_per_episode)
            order = permutations.get(episode_slot)
            if order is None:
                order = np.arange(self._samples_per_episode, dtype=np.int64)
                if self.shuffle:
                    np.random.default_rng(
                        self.seed + self.epoch * episode_count + episode_slot + 1
                    ).shuffle(order)
                permutations[episode_slot] = order
            start = batch_slot * self.batch_size
            stop = min(start + self.batch_size, self._samples_per_episode)
            if stop - start < self.batch_size and self.drop_last:
                continue
            yield (episode_slot * self._samples_per_episode + order[start:stop]).tolist()
