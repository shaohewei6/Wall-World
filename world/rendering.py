"""Pixel-identical local and overview rendering from the original environment."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

def render_viewport(self, position: np.ndarray | None = None) -> np.ndarray:
    """Render the ball-centred 224px crop; out-of-map pixels are walls."""
    center = self.position if position is None else np.asarray(position, dtype=np.float32)
    viewport = self.config.viewport_size
    half = viewport // 2
    left, top = int(round(center[0])) - half, int(round(center[1])) - half
    pixels = np.zeros((viewport, viewport, 3), dtype=np.uint8)
    source_x0, source_y0 = max(0, left), max(0, top)
    source_x1 = min(self.layout.map_size, left + viewport)
    source_y1 = min(self.layout.map_size, top + viewport)
    dest_x0, dest_y0 = source_x0 - left, source_y0 - top
    if source_x0 < source_x1 and source_y0 < source_y1:
        crop = self.layout.wall_mask[source_y0:source_y1, source_x0:source_x1]
        target = pixels[dest_y0 : dest_y0 + crop.shape[0], dest_x0 : dest_x0 + crop.shape[1]]
        target[~crop] = (255, 255, 255)
    ball_center = viewport // 2
    ball_half = self.config.ball_size // 2
    pixels[ball_center - ball_half : ball_center + ball_half, ball_center - ball_half : ball_center + ball_half] = (230, 40, 40)
    return pixels

def render_overview(self, *, scale: int = 4) -> np.ndarray:
    rgb = np.full((*self.layout.wall_mask.shape, 3), 255, dtype=np.uint8)
    rgb[self.layout.wall_mask] = (0, 0, 0)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    half = self.config.ball_size // 2
    x, y = self.position
    draw.rectangle((x - half, y - half, x + half - 1, y + half - 1), fill=(230, 40, 40))
    return np.asarray(
        image.resize((self.layout.map_size // scale, self.layout.map_size // scale), Image.Resampling.NEAREST)
    )
