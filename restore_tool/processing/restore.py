"""
restore_ac_v2.py

Restoration only.  Scratch detection is delegated to scratch.py.

The scratch mask returned by scratch.py already encodes the estimated
repair width for thin and soft scratches.  This file repairs only those pixels,
using row-wise left/right interpolation so that thin vertical scratches and soft
wide scratches do not become full-height inpaint bands.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .scratch import detect_scratch_segments_stack
from .dust import detect_dust


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _safe_bool_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != shape:
        m = cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _inpaint_mask(img: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    if not np.any(mask):
        return img.copy()
    result = cv2.inpaint(_to_u8(img), mask.astype(np.uint8) * 255, int(radius), cv2.INPAINT_TELEA)
    return result.astype(np.float32) / 255.0


# -----------------------------------------------------------------------------
# Scratch repair
# -----------------------------------------------------------------------------


def repair_vertical_mask_horizontal_interp(img: np.ndarray, mask: np.ndarray, p: Dict) -> np.ndarray:
    """
    Repair near-vertical scratch masks row-by-row.

    This is intentionally local and does not use full-height columns.  For each
    masked run on a row, the function samples clean pixels left/right of the
    run and interpolates across the defect.  It works for both thin and wide
    masks as long as scratch.py provides the correct width.
    """
    out = img.copy()
    mask = mask.astype(bool)
    if not np.any(mask):
        return out

    h, w = mask.shape
    radius = int(p.get("scratch_interp_radius", 42))
    sample = int(p.get("scratch_interp_sample", 8))
    max_run = int(p.get("scratch_interp_max_run", 80))

    for y in range(h):
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([xs[0]], xs[breaks + 1]))
        ends = np.concatenate((xs[breaks], [xs[-1]]))

        for x0, x1 in zip(starts, ends):
            run_w = int(x1 - x0 + 1)
            if run_w > max_run:
                # Something is wrong; do not smear a giant horizontal strip.
                continue

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


def restore_frame(frames: Sequence[np.ndarray], p: Dict, frame_idx=None, flow_cache=None):
    """
    Restore the center frame of a temporal stack.

    Args:
        frames: odd-length list of RGB float32 images in [0,1].
        p: parameter dictionary from app.py.
        frame_idx, flow_cache: accepted for compatibility with older app.py.
    """
    t0 = time.perf_counter()
    if not frames:
        raise ValueError("restore_frame requires at least one frame")

    c_idx = len(frames) // 2
    center = frames[c_idx].astype(np.float32)
    h, w = center.shape[:2]

    preset_mask = p.get("preset_mask", None)
    valid = _safe_bool_mask(preset_mask, (h, w))

    # --- Scratch detection ---
    t = time.perf_counter()
    scratch_mask, scratch_debug = detect_scratch_segments_stack(frames, p, preset_mask=preset_mask)
    scratch_mask = scratch_mask.astype(bool) & valid
    if p.get("debug_timing", False):
        print(f"[TIMING] scratch AC detection: {time.perf_counter() - t:.2f}s")

    # --- Optional dust detection ---
    # Dust repair is intentionally separate from scratches. It rejects scratch
    # halos and only repairs compact transient specks.
    dust_mask = np.zeros((h, w), dtype=bool)
    if p.get("enable_dust_repair", False):
        t = time.perf_counter()
        others = [frames[i].astype(np.float32) for i in range(len(frames)) if i != c_idx]
        try:
            dust_result = detect_dust(
                others,
                center,
                p,
                valid_mask=valid,
                scratch_mask=scratch_mask,
            )
            if isinstance(dust_result, tuple):
                dust_raw = np.zeros((h, w), dtype=bool)
                for part in dust_result:
                    if part is not None:
                        dust_raw |= (part > 0.5)
            else:
                dust_raw = dust_result > 0.5

            # Final safety: never let dust override scratch repairs.
            scratch_block = scratch_mask.copy()
            reject = int(p.get("dust_reject_scratch_dilate", 9))
            if reject > 0 and np.any(scratch_block):
                k = reject * 2 + 1
                scratch_block = cv2.dilate(
                    scratch_block.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                    iterations=1,
                ).astype(bool)

            dust_mask = dust_raw.astype(bool) & valid & ~scratch_block

        except Exception as e:
            print(f"[DUST WARNING] dust detection failed: {e}")

        if p.get("debug_timing", False):
            print(f"[TIMING] dust detection: {time.perf_counter() - t:.2f}s")

    # --- Repair scratches ---
    out = center.copy()
    if np.any(scratch_mask):
        interp = repair_vertical_mask_horizontal_interp(center, scratch_mask, p)
        strength = float(np.clip(p.get("scratch_interp_strength", 1.0), 0.0, 1.0))
        out[scratch_mask] = out[scratch_mask] * (1.0 - strength) + interp[scratch_mask] * strength

        if p.get("scratch_inpaint_after_interp", False):
            inp = _inpaint_mask(out, scratch_mask, radius=int(p.get("inpaint_radius", 3)))
            out[scratch_mask] = inp[scratch_mask]

    # --- Optional dust repair ---
    dust_only = dust_mask & ~scratch_mask
    if np.any(dust_only):
        inp = _inpaint_mask(out, dust_only, radius=int(p.get("dust_inpaint_radius", 3)))
        out[dust_only] = inp[dust_only]

    mask = (scratch_mask | dust_only).astype(np.float32)

    if p.get("debug_repair", False):
        diff = np.mean(np.abs(out - center), axis=2)
        print(
            f"[REPAIR DEBUG] scratch_px={int(np.sum(scratch_mask))}, "
            f"dust_px={int(np.sum(dust_only))}, changed_px={int(np.sum(diff > 1e-6))}, "
            f"max_diff={float(np.max(diff)):.6f}, mean_diff={float(np.mean(diff)):.6f}"
        )

    if p.get("debug_timing", False):
        print(f"[TIMING] total restore_frame: {time.perf_counter() - t0:.2f}s")

    return out.astype(np.float32), mask
