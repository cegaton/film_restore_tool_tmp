import numpy as np
import cv2
from .scratch import detect_scratches_stack

def restore_preview(c, p):
    scratch = detect_scratches_stack([c], p["scratch"], p)

    blurred = cv2.GaussianBlur(c, (5,5), 0)

    out = c*(1-scratch[...,None]) + blurred*(scratch[...,None])

    return out, scratch
