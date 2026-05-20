import numpy as np
import cv2

from .scratch import detect_scratches_stack
from .dust import detect_dust


def warp_to_center(ref, img):
    ref_gray = cv2.cvtColor((np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        img_gray,
        ref_gray,
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

    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return warped


def feather_mask(mask, radius=2):
    """
    Feather only very close to the detected defect.

    This prevents the repair from softly blending across large parts
    of the image.
    """
    m = mask.astype(np.float32)

    if radius <= 0:
        return m

    k = radius * 2 + 1

    support = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1
    ).astype(bool)

    blurred = cv2.GaussianBlur(m, (k, k), 0)

    blurred[~support] = 0.0

    return np.clip(blurred, 0.0, 1.0)


def inpaint_rgb(img, mask):
    mask_u8 = (mask.astype(np.uint8) * 255)

    src = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    result = cv2.inpaint(
        src,
        mask_u8,
        3,
        cv2.INPAINT_TELEA
    )

    return result.astype(np.float32) / 255.0
    
def refine_large_mask(mask, image, p):
    """
    Refine an overly large defect mask instead of skipping the frame.

    This is designed for noisy 16mm scans where dust/scratch detection may
    over-trigger on grain, texture, or scene detail.

    It keeps:
    - thin/vertical scratch-like components
    - compact dust-like blobs

    It removes:
    - huge regions
    - broad texture/noise detections
    - components covering too much of the frame
    """

    m = mask.astype(np.uint8)

    h, w = m.shape
    frame_area = h * w

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        m,
        connectivity=8
    )

    refined = np.zeros_like(m, dtype=np.uint8)

    max_component_fraction = p.get("max_component_fraction", 0.0025)
    max_component_area = frame_area * max_component_fraction

    min_component_area = p.get("min_component_area", 3)

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_component_area:
            continue

        # Reject massive blobs. These are usually image content or detector failure.
        if area > max_component_area:
            continue

        aspect = ch / max(1, cw)

        # Scratch-like: tall and narrow.
        is_scratch_like = (
            ch >= 25 and
            aspect >= 4.0 and
            cw <= 80
        )

        # Dust-like: compact-ish blob, not a huge region.
        fill_ratio = area / max(1, cw * ch)

        is_dust_like = (
            area <= max_component_area and
            cw <= 180 and
            ch <= 180 and
            fill_ratio >= 0.08
        )

        if is_scratch_like or is_dust_like:
            refined[labels == i] = 1

    # Gentle cleanup.
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    return refined.astype(bool)


def restore_frame(frames, p, frame_idx, flow_cache):
    n = len(frames)
    c_idx = n // 2
    c = frames[c_idx]

    # --- Scratch detection ---
    scratch = detect_scratches_stack(frames, p["scratch"], p)

    # --- Align neighboring frames ---
    aligned_frames = [None] * n

    for j in range(n):
        if j == c_idx:
            aligned_frames[j] = c
            continue

        key = (frame_idx, j)

        if key not in flow_cache:
            flow_cache[key] = warp_to_center(c, frames[j])

        aligned_frames[j] = flow_cache[key]

    others = [aligned_frames[j] for j in range(n) if j != c_idx]

    # --- Dust detection ---
    dust = detect_dust(others, c, p["dust"])

    # --- Combined damage mask ---
    mask = ((scratch > 0.5) | (dust > 0.5)).astype(np.float32)

    if not np.any(mask):
        return c, mask
    
    mask_bool = mask > 0
    
    # Safety check:
    # If too much of the frame is detected as damage, something is wrong.
    # Do not temporally blend large areas of the picture.
    max_fraction = p.get("max_mask_fraction", 0.015)
    mask_fraction = np.mean(mask_bool)
    
    if mask_fraction > max_fraction:
        print(
            f"WARNING: mask large on frame {frame_idx}: "
            f"{mask_fraction * 100:.2f}% detected. Refining mask instead of skipping."
        )
    
        mask_bool = refine_large_mask(mask_bool, c, p)
    
        mask_fraction_after = np.mean(mask_bool)
    
        print(
            f"Refined mask on frame {frame_idx}: "
            f"{mask_fraction_after * 100:.2f}% detected."
        )
    
        if not np.any(mask_bool):
            empty_mask = np.zeros(mask.shape, dtype=np.float32)
            return c, empty_mask
    
        mask = mask_bool.astype(np.float32)

    stack = np.stack(others, axis=0)
    temporal_median = np.median(stack, axis=0)
    temporal_std = np.std(stack, axis=0).mean(axis=2)

    out = c.copy()
    remaining = mask_bool.copy()

    # --- First pass: use best aligned neighboring pixel ---
    best_diff = np.full(mask_bool.shape, np.inf, dtype=np.float32)

    for j in range(n):
        if j == c_idx:
            continue

        aligned = aligned_frames[j]
        diff = np.linalg.norm(aligned - c, axis=2)

        improves = (
            remaining
            & (diff < best_diff)
            & (diff < 0.12)
            & (temporal_std < 0.10)
        )

        out[improves] = aligned[improves]
        best_diff[improves] = diff[improves]
        remaining[improves] = False

        if not np.any(remaining):
            break

    # --- Second pass: temporal median fallback ---
    # This is essential for large dust and wide scratches.
    if np.any(remaining):
        median_ok = remaining & (temporal_std < 0.14)
        out[median_ok] = temporal_median[median_ok]
        remaining[median_ok] = False

    # --- Final fallback: spatial inpaint ---
    # Only used where temporal replacement was not safe.
    if np.any(remaining):
        inpainted = inpaint_rgb(out, remaining)
        out[remaining] = inpainted[remaining]

    # --- Feather the repaired areas ---
    alpha = feather_mask(mask_bool, radius=1)
    out = c * (1.0 - alpha[..., None]) + out * alpha[..., None]

    return out.astype(np.float32), mask
