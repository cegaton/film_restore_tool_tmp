import numpy as np
import cv2

from .scratch import detect_scratches_stack


def restore_preview(c, p):
    scratch = detect_scratches_stack([c], p["scratch"], p)

    mask = scratch > 0.5

    out = c.copy()

    if np.any(mask):
        blurred = cv2.GaussianBlur(c, (7, 7), 0)
        out[mask] = blurred[mask]

    return out.astype(np.float32), scratch.astype(np.float32)
