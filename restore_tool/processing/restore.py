import numpy as np
import cv2
import time

from .scratch import detect_scratch_segments_stack
from .dust import detect_dust


def _to_u8(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _safe_bool_mask(mask, shape):
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    return m > 0.5


def repair_vertical_mask_horizontal_interp(img, mask, p):
    """
    Repair thin/near-vertical scratch masks row-by-row using left/right samples.

    This preserves local texture better than full-column inpainting and only
    touches pixels in the supplied mask.
    """
    out = img.copy()
    mask = mask.astype(bool)

    if not np.any(mask):
        return out

    h, w = mask.shape
    radius = int(p.get("scratch_interp_radius", 36))
    sample = int(p.get("scratch_interp_sample", 6))

    for y in range(h):
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([xs[0]], xs[breaks + 1]))
        ends = np.concatenate((xs[breaks], [xs[-1]]))

        for x0, x1 in zip(starts, ends):
            left = np.arange(max(0, x0 - radius), x0)
            right = np.arange(x1 + 1, min(w, x1 + 1 + radius))

            left = left[~mask[y, left]]
            right = right[~mask[y, right]]

            if left.size > sample:
                left = left[-sample:]
            if right.size > sample:
                right = right[:sample]

            has_l = left.size > 0
            has_r = right.size > 0
            if not has_l and not has_r:
                continue

            fill_x = np.arange(x0, x1 + 1)

            if has_l:
                lc = np.mean(img[y, left], axis=0)
            if has_r:
                rc = np.mean(img[y, right], axis=0)

            if has_l and has_r:
                t = ((fill_x - x0) / max(1, x1 - x0))[:, None]
                fill = lc * (1.0 - t) + rc * t
            elif has_l:
                fill = np.repeat(lc[None, :], fill_x.size, axis=0)
            else:
                fill = np.repeat(rc[None, :], fill_x.size, axis=0)

            out[y, fill_x] = fill

    return out


def inpaint_mask(img, mask, radius=3):
    if not np.any(mask):
        return img.copy()
    result = cv2.inpaint(
        _to_u8(img),
        mask.astype(np.uint8) * 255,
        int(radius),
        cv2.INPAINT_TELEA,
    )
    return result.astype(np.float32) / 255.0


def restore_frame(frames, p, frame_idx=None, flow_cache=None):
    """
    Restore the center frame of a temporal stack.

    Detection is delegated to scratch.py. This function only repairs the masks.
    """
    t0 = time.perf_counter()

    if not frames:
        raise ValueError("restore_frame requires at least one frame")

    c_idx = len(frames) // 2
    center = frames[c_idx].astype(np.float32)
    h, w = center.shape[:2]

    preset_mask = p.get("preset_mask", None)
    valid = _safe_bool_mask(preset_mask, (h, w))

    # --- Scratch detection from scratch.py ---
    t = time.perf_counter()
    scratch_mask, scratch_debug = detect_scratch_segments_stack(
        frames,
        p,
        preset_mask=preset_mask,
    )
    scratch_mask &= valid

    if p.get("debug_timing", False):
        print(f"[TIMING] scratch segment detection: {time.perf_counter() - t:.2f}s")

    # --- Optional dust detection, off by default during scratch tuning ---
    dust_mask = np.zeros((h, w), dtype=bool)
    if p.get("enable_dust_repair", False):
        t = time.perf_counter()
        others = [frames[i].astype(np.float32) for i in range(len(frames)) if i != c_idx]
        try:
            dust_result = detect_dust(others, center, float(p.get("dust", 0.045)))
            if isinstance(dust_result, tuple):
                d0 = dust_result[0]
            else:
                d0 = dust_result
            dust_mask = (d0 > 0.5) & valid
        except Exception as e:
            print(f"[DUST WARNING] dust detection failed: {e}")
            dust_mask = np.zeros((h, w), dtype=bool)
        if p.get("debug_timing", False):
            print(f"[TIMING] dust detection: {time.perf_counter() - t:.2f}s")

    # --- Repair scratches ---
    out = center.copy()

    if np.any(scratch_mask):
        interp = repair_vertical_mask_horizontal_interp(center, scratch_mask, p)
        strength = float(np.clip(p.get("scratch_interp_strength", 1.0), 0.0, 1.0))
        out[scratch_mask] = out[scratch_mask] * (1.0 - strength) + interp[scratch_mask] * strength

        # Optional tiny inpaint pass only inside the same mask, useful when left/right samples fail.
        if p.get("scratch_inpaint_after_interp", False):
            inpainted = inpaint_mask(out, scratch_mask, radius=int(p.get("inpaint_radius", 3)))
            out[scratch_mask] = inpainted[scratch_mask]

    # --- Repair dust separately with small inpaint ---
    dust_only = dust_mask & ~scratch_mask
    if np.any(dust_only):
        inpainted = inpaint_mask(out, dust_only, radius=int(p.get("dust_inpaint_radius", 3)))
        out[dust_only] = inpainted[dust_only]

    mask = (scratch_mask | dust_only).astype(np.float32)

    if p.get("debug_repair", False):
        diff = np.mean(np.abs(out - center), axis=2)
        changed = int(np.sum(diff > 1e-6))
        print(
            f"[REPAIR DEBUG] scratch_px={int(np.sum(scratch_mask))}, "
            f"dust_px={int(np.sum(dust_only))}, changed_px={changed}, "
            f"max_diff={float(np.max(diff)):.6f}, mean_diff={float(np.mean(diff)):.6f}"
        )

    if p.get("debug_timing", False):
        print(f"[TIMING] total restore_frame: {time.perf_counter() - t0:.2f}s")

    return out.astype(np.float32), mask
