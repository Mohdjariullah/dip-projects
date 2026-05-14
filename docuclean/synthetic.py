"""Synthetic degraded-document generator for DocuClean.

Produces a page of fake text with realistic degradation: uneven illumination,
yellow coffee-stain blobs, photocopy noise, and faded ink. Designed so Otsu
collapses, but Sauvola/Niblack recover cleanly.
"""

from __future__ import annotations

import cv2
import numpy as np


def _draw_text_block(canvas: np.ndarray, top: int, lines: list[str],
                     scale: float = 0.6, ink: int = 60) -> None:
    """Draw faded text lines starting at y=top."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = top
    for line in lines:
        cv2.putText(canvas, line, (40, y), font, scale, (ink, ink, ink), 1, cv2.LINE_AA)
        y += int(28 * scale * 1.6)


def make_faded_document(width: int = 600, height: int = 800,
                        shading: float = 0.6, coffee: bool = True,
                        noise: float = 6.0, seed: int | None = 1) -> np.ndarray:
    """Faded multi-line document on a yellowed page. Returns 3-channel BGR."""
    rng = np.random.default_rng(seed)
    paper = np.full((height, width, 3), 220, dtype=np.uint8)
    paper[..., 0] = 200   # B
    paper[..., 1] = 215   # G — slight yellow tint
    paper[..., 2] = 220   # R

    canvas = paper.copy()

    lines = [
        "DocuClean Pro - degraded document sample",
        "Adaptive thresholding for low-contrast text.",
        "Otsu fails on uneven illumination; Sauvola",
        "and Niblack adapt the threshold per window.",
        "",
        "The quick brown fox jumps over the lazy dog.",
        "Sphinx of black quartz, judge my vow.",
        "0 1 2 3 4 5 6 7 8 9    abcdef ghijkl mnop",
        "",
        "Local statistics make this robust to faded ink,",
        "yellow coffee stains, and photocopy banding.",
    ]
    _draw_text_block(canvas, top=80, lines=lines, scale=0.7, ink=90)

    # Coffee-stain blobs (yellow-brown ellipses, blurred)
    if coffee:
        stain = np.zeros_like(canvas)
        cv2.ellipse(stain, (int(width * 0.7), int(height * 0.25)),
                    (140, 80), 30, 0, 360, (10, 70, 130), -1)
        cv2.ellipse(stain, (int(width * 0.3), int(height * 0.78)),
                    (110, 70), -20, 0, 360, (20, 80, 140), -1)
        stain = cv2.GaussianBlur(stain, (61, 61), 0)
        canvas = cv2.addWeighted(canvas, 1.0, stain, -0.5, 0)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # Uneven illumination — radial darkening to the corners
    if shading > 0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
        r /= r.max()
        shade = 1.0 - shading * r
        canvas = (canvas.astype(np.float32) * shade[..., None]).astype(np.uint8)

    # Photocopy / sensor noise
    if noise > 0:
        canvas = np.clip(
            canvas.astype(np.float32) + rng.normal(0, noise, canvas.shape),
            0, 255,
        ).astype(np.uint8)

    return canvas


def make_low_contrast_doc(width: int = 600, height: int = 800, seed: int | None = 2) -> np.ndarray:
    """Doc with very faint ink (low contrast). Demonstrates contrast-stretch + Sauvola."""
    return make_faded_document(width, height, shading=0.2, coffee=False,
                               noise=3.0, seed=seed)


if __name__ == "__main__":
    cv2.imwrite("samples/faded_doc.png", make_faded_document())
    cv2.imwrite("samples/lowcontrast_doc.png", make_low_contrast_doc())
    print("Wrote samples/faded_doc.png, lowcontrast_doc.png")
