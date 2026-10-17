"""Original atomic patch movement and collision semantics."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from .map import LargeRoomLayout, LargeRoomWorldConfig, PATCH_ACTIONS, PATCH_ACTION_DISPLACEMENTS
from .rendering import render_viewport, render_overview

class LargeRoomEnvironment:
    """Square-agent dynamics and fixed local observation for the large layout."""

    def __init__(self, layout: LargeRoomLayout, config: LargeRoomWorldConfig) -> None:
        if layout.map_size != config.map_size:
            raise ValueError("layout and config map sizes disagree")
        self.layout = layout
        self.config = config
        self.position = np.zeros(2, dtype=np.float32)

    def is_free(self, position: np.ndarray) -> bool:
        """Check exact overlap between the 14px square and true wall pixels.

        A square may touch a wall edge.  Only a candidate whose rendered
        14x14 footprint overlaps black wall pixels is a collision.  This gives
        discrete patch actions the intuitive rule: an adjacent free patch is
        reachable, while entering a wall patch is not.
        """
        x, y = (int(round(value)) for value in np.asarray(position, dtype=np.float32).reshape(2))
        half = self.config.ball_size // 2
        left, top = x - half, y - half
        right, bottom = left + self.config.ball_size, top + self.config.ball_size
        if left < 0 or top < 0 or right > self.layout.map_size or bottom > self.layout.map_size:
            return False
        return not bool(self.layout.wall_mask[top:bottom, left:right].any())

    def sample_free_position(self, rng: np.random.Generator) -> np.ndarray:
        rows, columns = np.nonzero(~self.layout.wall_mask)
        for _ in range(512):
            index = int(rng.integers(len(rows)))
            candidate = np.array([columns[index], rows[index]], dtype=np.float32)
            if self.is_free(candidate):
                return candidate
        raise RuntimeError("Could not sample a collision-free square position")

    def reset(self, position: np.ndarray | None = None, *, seed: int | None = None) -> np.ndarray:
        candidate = self.sample_free_position(np.random.default_rng(seed)) if position is None else np.asarray(position, dtype=np.float32)
        if not self.is_free(candidate):
            raise ValueError("Initial square center is not collision-free")
        self.position = candidate.copy()
        return self.render_viewport()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(command))
        if norm > self.config.max_action_norm:
            command *= self.config.max_action_norm / norm
            norm = self.config.max_action_norm
        substeps = max(1, int(np.ceil(norm / self.config.collision_substep)))
        increment = command / substeps
        start = self.position.copy()
        collision = False
        for _ in range(substeps):
            proposal = self.position + increment
            if not self.is_free(proposal):
                collision = True
                break
            self.position = proposal.astype(np.float32)
        return self.render_viewport(), self.position.copy(), {
            "collision": collision,
            "commanded_displacement": command.copy(),
            "actual_displacement": self.position - start,
        }

    def step_patch(self, action_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Atomically execute one of the eight adjacent 14px patch moves.

        Unlike continuous ``step()``, a discrete patch move never stops partway
        through a blocked move: the agent either reaches the neighbouring patch
        center or remains exactly on its current patch center.
        """
        index = int(action_id)
        if not 0 <= index < len(PATCH_ACTIONS):
            raise ValueError(f"action_id must be in [0, {len(PATCH_ACTIONS) - 1}]")
        command = PATCH_ACTION_DISPLACEMENTS[index].astype(np.float32)*self.config.patch_size
        start = self.position.copy()
        proposal = start + command
        collision = not self.is_free(proposal)
        if not collision:
            self.position = proposal
        return self.render_viewport(), self.position.copy(), {
            "collision": collision,
            "commanded_displacement": command,
            "actual_displacement": self.position - start,
        }

    render_viewport = render_viewport

    render_overview = render_overview

class PatchAgentEnvironment:
    """Expose LargeRoomEnvironment using integer patch coordinates only."""
    def __init__(self,environment: LargeRoomEnvironment) -> None:
        self.environment = environment
        self.grid_size = environment.layout.map_size//environment.config.patch_size
        if environment.layout.map_size%environment.config.patch_size:
            raise ValueError("map size must be divisible by patch size")
        self._free_positions: np.ndarray | None = None
    def _patch_position(self,value: Any,name: str) -> np.ndarray:
        position = np.asarray(value)
        if position.shape != (2,):
            raise ValueError(f"{name} must have shape [2]")
        if not np.issubdtype(position.dtype,np.integer):
            raise TypeError(f"{name} must use integer patch coordinates")
        position = position.astype(np.int64,copy=True)
        if (position<0).any() or (position>=self.grid_size).any():
            raise ValueError(f"{name} must lie in [0,{self.grid_size-1}]")
        return position
    def _to_environment_position(self,patch_position: np.ndarray) -> np.ndarray:
        return patch_position.astype(np.float32)*self.environment.config.patch_size+self.environment.config.patch_size/2
    def _from_environment_position(self,position: np.ndarray) -> np.ndarray:
        patch_position=(np.asarray(position,dtype=np.float32)-self.environment.config.patch_size/2)/self.environment.config.patch_size
        rounded=np.rint(patch_position)
        if not np.allclose(patch_position,rounded):
            raise RuntimeError("environment position is not aligned to the patch lattice")
        return self._patch_position(rounded.astype(np.int64),"environment position")
    @property
    def position(self) -> np.ndarray:
        return self._from_environment_position(self.environment.position)
    def _sample_free_patch_position(self,rng: np.random.Generator) -> np.ndarray:
        if self._free_positions is None:
            free=[]
            for patch_y in range(self.grid_size):
                for patch_x in range(self.grid_size):
                    candidate=np.asarray((patch_x,patch_y),dtype=np.int64)
                    if self.environment.is_free(self._to_environment_position(candidate)):
                        free.append(candidate)
            if not free:
                raise RuntimeError("no collision-free patch position is available")
            self._free_positions=np.stack(free)
        return self._free_positions[int(rng.integers(len(self._free_positions)))].copy()
    def reset(self,position: np.ndarray | None = None,*,seed: int | None = None) -> tuple[np.ndarray,np.ndarray]:
        if position is None and getattr(self, "default_start", None) is not None:
            position = self.default_start
        if seed is None:
            seed = getattr(self, "default_seed", None)
        patch_position=self._sample_free_patch_position(np.random.default_rng(seed)) if position is None else self._patch_position(position,"position")
        observation=self.environment.reset(self._to_environment_position(patch_position))
        return observation,self.position
    def step(self,action_id: int) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
        if not isinstance(action_id,(int,np.integer)):
            raise TypeError("action_id must be an integer")
        action_id=int(action_id)
        if not 0<=action_id<len(PATCH_ACTION_DISPLACEMENTS):
            raise ValueError(f"action_id must lie in [0,{len(PATCH_ACTION_DISPLACEMENTS)-1}]")
        start=self.position
        observation,environment_position,environment_info=self.environment.step_patch(action_id)
        position=self._from_environment_position(environment_position)
        commanded_displacement=PATCH_ACTION_DISPLACEMENTS[action_id].astype(np.int64)
        actual_displacement=position-start
        collision=bool(environment_info["collision"])
        if collision and np.any(actual_displacement):
            raise RuntimeError("collided patch action changed patch position")
        if not collision and not np.array_equal(actual_displacement,commanded_displacement):
            raise RuntimeError("successful patch action did not move by one patch")
        return observation,position,{"collision":collision,"commanded_displacement":commanded_displacement,"actual_displacement":actual_displacement}

    def navigation_goal(self, origin=None):
        """Convert configured goal outside Agent; call reset before using current origin."""
        task = getattr(self, "task", None)
        if task is None or task['goal'] is None:
            raise ValueError('No goal configured in world.task; supply a goal explicitly')
        origin = self.position if origin is None else self._patch_position(origin, 'origin')
        goal = np.asarray(task['goal'], dtype=np.int64)
        absolute = goal if task['coordinate_mode']=='absolute_patch' else origin+goal
        absolute = self._patch_position(absolute, 'goal')
        if not self.environment.is_free(self._to_environment_position(absolute)):
            raise ValueError('Configured goal is blocked')
        return absolute-origin
