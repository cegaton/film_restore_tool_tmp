import numpy as np
import cv2


def detect_scratches_stack(frames, thresh, p):
    """
    Vectorized scratch detection across all frames
    Returns a single combined mask
    """

    # --- stack frames ---
    stack = np.stack(frames, axis=0)  # (N, H, W, 3)

    # --- grayscale ---
    gray = np.mean(stack, axis=3)     # (N, H, W)

    # normalize (per frame)
    std = np.std(gray, axis=(1, 2), keepdims=True) + 1e-6
    gray_norm = gray / std

    # --- morphological response (per frame, vectorized loop over kernel sizes only) ---
    scales = [9, 21, 41]
    responses = []

    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, s))

        # apply per frame (OpenCV not batch-aware → minimal loop here)
        resp = np.stack([
            cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
            for g in gray_norm
        ], axis=0)

        responses.append(resp)

    # --- combine scales ---
    resp = np.maximum.reduce(responses)

    # --- threshold ---
    masks = (resp > thresh).astype(np.uint8)
    
    # --- temporal voting ---
    votes = np.sum(masks, axis=0)
    
    # --- voting rule ---
    if len(frames) == 1:
        votes_required = 1
    else:
        votes_required = max(2, len(frames)//2)
    
    # --- final mask ---
    mask = (votes >= votes_required).astype(np.float32)
    
    return mask
