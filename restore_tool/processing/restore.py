import numpy as np
import cv2
from .scratch import detect_scratches_stack
from .dust import detect_dust


def warp_to_center(ref, img):
    """
    Warp img to align with ref using optical flow
    """
    ref_gray = cv2.cvtColor((ref*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        img_gray, ref_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = flow.shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return warped

def restore_frame(frames, p, frame_idx, flow_cache):
    n = len(frames)
    c_idx = n // 2
    c = frames[c_idx]

    from .scratch import detect_scratches_stack
    
    scratch = detect_scratches_stack(frames, p["scratch"], p)

    # --- ALIGN ALL FRAMES ONCE ---
    aligned_frames = [None] * n
    
    for j in range(n):
        if j == c_idx:
            aligned_frames[j] = c
            continue
    
        key = (frame_idx, j)
    
        if key not in flow_cache:
            flow_cache[key] = warp_to_center(c, frames[j])
    
        aligned_frames[j] = flow_cache[key]
    
    # --- DUST ---
    others = [aligned_frames[j] for j in range(n) if j != c_idx]
    dust = detect_dust(others, c, p["dust"])

    # --- MASK (FIXED) ---
    mask = ((scratch > 0.5) | (dust > 0.5)).astype(np.float32)

    if not np.any(mask):
        return c, mask

    mask_bool = mask > 0

    # --- TEMPORAL STABILITY (FIXED) ---
    stack = np.stack([f for i,f in enumerate(aligned_frames) if i != c_idx], axis=0)
    std = np.std(stack, axis=0).mean(axis=2)

    # --- REPLACEMENT ---
    out = c.copy()
    remaining = mask_bool.copy()

    best_diff = np.full(mask_bool.shape, np.inf, dtype=np.float32)

    for j in range(n):
        if j == c_idx:
            continue

        aligned = aligned_frames[j]
        diff = np.linalg.norm(aligned - c, axis=2)

        improves = (
            (diff < best_diff) &
            (diff < 0.05) &
            (std < 0.03) &
            remaining
        )

        out[improves] = aligned[improves]
        best_diff[improves] = diff[improves]
        remaining[improves] = False

        if not np.any(remaining):
            break

    # --- FALLBACK (CRITICAL) ---
    out[remaining] = c[remaining]

    return out, mask
