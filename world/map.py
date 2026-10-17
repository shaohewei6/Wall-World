from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from pathlib import Path
if "." in __package__:
    from ..util.config import validate_geometry
else:
    from util.config import validate_geometry


def load_layout(path):
    """Read occupancy only; trajectory RGB arrays are not loaded into memory."""
    with np.load(Path(path), allow_pickle=False) as archive:
        mask = archive["wall_mask"].astype(bool)
    return LargeRoomLayout(mask, {})

# Eight neighbouring patch moves in the historical checkpoint's exact order.
PATCH_ACTION_DISPLACEMENTS = np.asarray(
    (
        (0,-1),
        (0,1),
        (-1,0),
        (1,0),
        (-1,-1),
        (1,-1),
        (-1,1),
        (1,1),
    ),
    dtype=np.int8,
)
# Backward-compatible environment-internal displacements.
PATCH_ACTIONS = PATCH_ACTION_DISPLACEMENTS.astype(np.int16)*14


@dataclass(frozen=True)
class LargeRoomWorldConfig:
    """Fixed observation geometry and randomized rectangular-room limits."""

    viewport_size: int = 224
    patch_size: int = 14
    map_tiles_per_axis: int = 4
    tile_size: int = 224
    wall_thickness: int = 28
    ball_size: int = 14
    min_room_count: int = 7
    max_room_count: int = 7
    min_room_width: int = 182
    max_room_width: int = 294
    min_room_height: int = 168
    max_room_height: int = 252
    room_clearance: int = 28
    min_doors_per_room: int = 1
    max_doors_per_room: int = 4
    door_width: int = 56
    max_placement_attempts: int = 240
    max_generation_attempts: int = 40
    max_action_norm: float = 20.0
    collision_substep: float = 1.0

    def __post_init__(self) -> None:
        validate_geometry(asdict(self))

    @property
    def map_size(self) -> int:
        return self.map_tiles_per_axis * self.tile_size

@dataclass
class LargeRoomLayout:
    """A binary large-map occupancy mask and reproducible room metadata."""

    wall_mask: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.wall_mask = np.asarray(self.wall_mask, dtype=bool)
        if self.wall_mask.ndim != 2 or self.wall_mask.shape[0] != self.wall_mask.shape[1]:
            raise ValueError("wall_mask must be a square [H, W] array")

    @property
    def map_size(self) -> int:
        return int(self.wall_mask.shape[0])

class LargeRoomLayoutGenerator:
    """Generate rectangular frame rooms at random non-overlapping positions.

    Every room occupies a unique axis-aligned bounding rectangle, expanded by a
    configurable clearance during placement.  Its rectangular wall frame is
    then given a random count of random-width doors, with at least one door per
    room.  The surrounding map remains open navigable space.
    """

    def __init__(self, config: LargeRoomWorldConfig = LargeRoomWorldConfig()) -> None:
        self.config = config

    def sample(self, seed: int) -> LargeRoomLayout:
        rng = np.random.default_rng(int(seed))
        for generation_attempt in range(self.config.max_generation_attempts):
            room_count = int(
                rng.integers(self.config.min_room_count, self.config.max_room_count + 1)
            )
            rooms = self._place_rooms(rng, room_count)
            if rooms is not None:
                return self._rasterize(rooms, rng, int(seed), generation_attempt)
        raise RuntimeError("Could not place the requested number of non-overlapping rooms")

    def _place_rooms(
        self, rng: np.random.Generator, room_count: int
    ) -> list[tuple[int, int, int, int]] | None:
        config = self.config
        rooms: list[tuple[int, int, int, int]] = []
        margin = config.wall_thickness + config.room_clearance
        for _ in range(room_count):
            accepted = False
            for _ in range(config.max_placement_attempts):
                width = int(rng.integers(config.min_room_width // config.patch_size, config.max_room_width // config.patch_size + 1)) * config.patch_size
                height = int(rng.integers(config.min_room_height // config.patch_size, config.max_room_height // config.patch_size + 1)) * config.patch_size
                x0 = int(rng.integers(margin // config.patch_size, (config.map_size - margin - width) // config.patch_size + 1)) * config.patch_size
                y0 = int(rng.integers(margin // config.patch_size, (config.map_size - margin - height) // config.patch_size + 1)) * config.patch_size
                candidate = (x0, y0, x0 + width, y0 + height)
                if any(self._overlaps_with_clearance(candidate, room) for room in rooms):
                    continue
                rooms.append(candidate)
                accepted = True
                break
            if not accepted:
                return None
        return rooms

    def _overlaps_with_clearance(
        self, first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> bool:
        padding = self.config.room_clearance
        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second
        return not (
            ax1 + padding <= bx0
            or bx1 + padding <= ax0
            or ay1 + padding <= by0
            or by1 + padding <= ay0
        )

    def _rasterize(
        self,
        rooms: list[tuple[int, int, int, int]],
        rng: np.random.Generator,
        seed: int,
        generation_attempt: int,
    ) -> LargeRoomLayout:
        config = self.config
        mask = np.zeros((config.map_size, config.map_size), dtype=bool)
        thickness = config.wall_thickness
        mask[:thickness, :] = True
        mask[-thickness:, :] = True
        mask[:, :thickness] = True
        mask[:, -thickness:] = True
        room_records: list[dict[str, Any]] = []
        doors: list[dict[str, Any]] = []
        for room_id, bounds in enumerate(rooms):
            x0, y0, x1, y1 = bounds
            mask[y0 : y0 + thickness, x0:x1] = True
            mask[y1 - thickness : y1, x0:x1] = True
            mask[y0:y1, x0 : x0 + thickness] = True
            mask[y0:y1, x1 - thickness : x1] = True
            door_count = int(
                rng.integers(config.min_doors_per_room, config.max_doors_per_room + 1)
            )
            room_doors = [self._carve_door(mask, bounds, rng, room_id, len(doors) + index) for index in range(door_count)]
            doors.extend(room_doors)
            room_records.append(
                {
                    "room_id": room_id,
                    "bounds": [x0, y0, x1, y1],
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "door_count": door_count,
                    "door_ids": [door["door_id"] for door in room_doors],
                }
            )
        metadata = {
            "schema_version": 2,
            "seed": seed,
            "generation_attempt": generation_attempt,
            "generator": "large_random_rectangular_rooms_v2",
            "config": asdict(config),
            "map_size": config.map_size,
            "map_native_tiles": [config.map_tiles_per_axis, config.map_tiles_per_axis],
            "room_count": len(room_records),
            "wall_thickness": thickness,
            "ball_shape": "square",
            "ball_size": config.ball_size,
            "viewport_size": config.viewport_size,
            "door_count": len(doors),
            "rooms": room_records,
            "doors": doors,
            "free_fraction": float((~mask).mean()),
        }
        return LargeRoomLayout(mask, metadata)

    def _carve_door(
        self,
        mask: np.ndarray,
        bounds: tuple[int, int, int, int],
        rng: np.random.Generator,
        room_id: int,
        door_id: int,
    ) -> dict[str, Any]:
        x0, y0, x1, y1 = bounds
        config = self.config
        width = config.door_width
        half = width // 2
        side = str(rng.choice(("top", "right", "bottom", "left")))
        if side in {"top", "bottom"}:
            center_x = int(rng.integers((x0 + config.wall_thickness + half) // config.patch_size, (x1 - config.wall_thickness - half) // config.patch_size + 1)) * config.patch_size
            center_y = y0 if side == "top" else y1
            y_slice = slice(y0, y0 + config.wall_thickness) if side == "top" else slice(y1 - config.wall_thickness, y1)
            mask[y_slice, center_x - half : center_x + half] = False
            center = [center_x, center_y]
            orientation = "horizontal_wall"
        else:
            center_y = int(rng.integers((y0 + config.wall_thickness + half) // config.patch_size, (y1 - config.wall_thickness - half) // config.patch_size + 1)) * config.patch_size
            center_x = x1 if side == "right" else x0
            x_slice = slice(x1 - config.wall_thickness, x1) if side == "right" else slice(x0, x0 + config.wall_thickness)
            mask[center_y - half : center_y + half, x_slice] = False
            center = [center_x, center_y]
            orientation = "vertical_wall"
        return {
            "door_id": door_id,
            "room_id": room_id,
            "side": side,
            "orientation": orientation,
            "center": center,
            "width": width,
        }
