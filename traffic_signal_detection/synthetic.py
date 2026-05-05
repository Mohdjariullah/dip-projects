"""Synthetic traffic-light scene generator.

Used to guarantee a working demo when no real photo is available.
Each scene is a simple BGR image with one or more traffic-light housings
on a gray "street" background, plus optional Gaussian sensor noise.
"""

from __future__ import annotations

import numpy as np
import cv2


_LIGHT_COLORS = {
    "red":    (40, 40, 230),
    "yellow": (40, 230, 230),
    "green":  (60, 220, 60),
}
_DIM = (40, 40, 40)  # unlit bulb color


def _draw_light(canvas: np.ndarray, top_left: tuple, state: str, radius: int = 22) -> None:
    """Draw a single 3-bulb traffic-light housing onto the canvas in-place."""
    x, y = top_left
    # Housing
    pad = 8
    housing_w = radius * 2 + pad * 2
    housing_h = radius * 6 + pad * 4
    cv2.rectangle(
        canvas,
        (x, y),
        (x + housing_w, y + housing_h),
        (15, 15, 15),
        thickness=-1,
    )

    # Bulbs (top to bottom: red, yellow, green)
    cx = x + housing_w // 2
    centers = [
        (cx, y + pad + radius),                  # red
        (cx, y + pad + radius * 3 + pad // 2),   # yellow
        (cx, y + pad + radius * 5 + pad),        # green
    ]
    for bulb_state, center in zip(("red", "yellow", "green"), centers):
        color = _LIGHT_COLORS[bulb_state] if bulb_state == state else _DIM
        cv2.circle(canvas, center, radius, color, thickness=-1)
        # Soft glow on the lit bulb to mimic emission
        if bulb_state == state:
            glow = canvas.copy()
            cv2.circle(glow, center, radius + 6, color, thickness=-1)
            canvas[:] = cv2.addWeighted(canvas, 0.85, glow, 0.15, 0)


def make_traffic_scene(
    state: str | list = "red",
    width: int = 640,
    height: int = 480,
    noise: float = 0.04,
    seed: int | None = None,
) -> np.ndarray:
    """Render a traffic-light scene.

    Args:
        state: "red" | "yellow" | "green", OR a list of states to draw multiple
               lights at evenly spaced positions.
        width, height: output image size in pixels.
        noise: Gaussian noise std-dev as a fraction of 255.
        seed: optional RNG seed for reproducibility.

    Returns:
        BGR uint8 ndarray.
    """
    rng = np.random.default_rng(seed)

    # Gray "street" background with a slight vertical gradient.
    grad = np.linspace(110, 160, height, dtype=np.float32)[:, None]
    canvas = np.tile(grad, (1, width))
    canvas = cv2.merge([canvas, canvas, canvas]).astype(np.uint8)

    # Add a "road" rectangle at the bottom for context.
    cv2.rectangle(canvas, (0, int(height * 0.7)), (width, height), (60, 60, 60), -1)

    states = [state] if isinstance(state, str) else list(state)
    n = len(states)
    for i, s in enumerate(states):
        slot_w = width // (n + 1)
        x = slot_w * (i + 1) - 30
        y = int(height * 0.15)
        _draw_light(canvas, (x, y), s)

    if noise > 0:
        noise_layer = rng.normal(0, noise * 255, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise_layer, 0, 255).astype(np.uint8)

    return canvas


def make_blank_street(width: int = 640, height: int = 480, noise: float = 0.04) -> np.ndarray:
    """Background with no traffic lights (edge-case test for the detector)."""
    return make_traffic_scene(state=[], width=width, height=height, noise=noise)


if __name__ == "__main__":
    # Quick standalone sanity-run.
    cv2.imwrite("samples/red.png", make_traffic_scene("red", seed=1))
    cv2.imwrite("samples/yellow.png", make_traffic_scene("yellow", seed=2))
    cv2.imwrite("samples/green.png", make_traffic_scene("green", seed=3))
    cv2.imwrite("samples/multi.png", make_traffic_scene(["red", "green"], seed=4))
    print("Wrote samples/red.png, yellow.png, green.png, multi.png")
