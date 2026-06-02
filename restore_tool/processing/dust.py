"""
dust_ac_v1.py

Conservative dust/speck detector for the AC restoration pipeline.

Design goals:
- Detect compact bright/dark dust specks.
- Do NOT repair vertical scratches here; scratch.py owns line defects.
- Use temporal difference as a safety gate when neighboring frames exist.
- Use component geometry to reject image details, highlights, and edges.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.ndim == 2:
        return np.clip(img, 0.0, 1.0)
    # Use max channel slightly favors white/colored dust on negative scans.
    return np.clip(0.45 * np.mean(img, axis=2) + 0.55 * np.max(img, axis=2), 0.0, 1.0)


def _odd(v: int, min_v: int = 3) -> int:
    v = int(max(min_v, v))
    return v if v % 2 == 1 else v + 1


def _as_bool_mask(mask: Optional[np.ndarray], shape: Tuple[int, int], default: bool = True) -> np.ndarray:
    if mask is None:
        return np.full(shape, default, dtype=bool)
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != shape:
        m = cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _local_noise(gray: np.ndarray, bg: np.ndarray, box: int) -> np.ndarray:
    # Fast local MAD-like estimate.  Not a true median, but stable and cheap.
    resid = np.abs(gray - bg).astype(np.float32)
    box = _odd(box, 15)
    noise = cv2.blur(resid, (box, box))
    return np.maximum(noise * 1.4826, 1e-5).astype(np.float32)


def _temporal_outlier(
    gray: np.ndarray,
    other_grays: Sequence[np.ndarray],
    p: Dict,
) -> np.ndarray:
    if not other_grays:
        return np.ones(gray.shape, dtype=bool)

    stack = np.stack(other_grays, axis=0).astype(np.float32)
    med = np.median(stack, axis=0)
    std = np.std(stack, axis=0)

    diff = np.abs(gray - med)

    # Blur diff a little so subpixel/frame grain does not dominate.
    br = int(p.get("dust_temporal_blur", 1))
    if br > 0:
        k = br * 2 + 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)

    temporal_abs = float(p.get("dust_temporal_abs", 0.018))
    max_neighbor_std = float(p.get("dust_temporal_max_std", 0.10))

    return (diff >= temporal_abs) & (std <= max_neighbor_std)


def _component_filter(mask: np.ndarray, response: np.ndarray, valid: np.ndarray, scratch_mask: np.ndarray, p: Dict) -> np.ndarray:
    h, w = mask.shape
    frame_area = h * w

    m = (mask & valid).astype(np.uint8)

    # Never let dust repair eat line scratches or scratch halos.
    reject_scratch = int(p.get("dust_reject_scratch_dilate", 9))
    if np.any(scratch_mask) and reject_scratch > 0:
        k = reject_scratch * 2 + 1
        scratch_block = cv2.dilate(
            scratch_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        ).astype(bool)
        m[scratch_block] = 0

    # Remove tiny isolated hot pixels before labeling if desired.
    open_size = int(p.get("dust_open_size", 0))
    if open_size > 1:
        k = _odd(open_size, 3)
        m = cv2.morphologyEx(
            m,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )

    close_size = int(p.get("dust_close_size", 3))
    if close_size > 1:
        k = _odd(close_size, 3)
        m = cv2.morphologyEx(
            m,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m, dtype=np.uint8)

    min_area = int(p.get("dust_min_area", 3))
    max_area = int(p.get("dust_max_area", 650))
    max_w = int(p.get("dust_max_width", 46))
    max_h = int(p.get("dust_max_height", 46))
    max_aspect = float(p.get("dust_max_aspect", 2.8))
    min_fill = float(p.get("dust_min_fill_ratio", 0.12))
    min_peak = float(p.get("dust_min_peak_response", 0.025))
    min_mean = float(p.get("dust_min_mean_response", 0.010))

    kept = 0
    rejected = 0

    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            rejected += 1
            continue
        if cw > max_w or ch > max_h:
            rejected += 1
            continue

        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        if aspect > max_aspect:
            rejected += 1
            continue

        fill = area / max(1, cw * ch)
        if fill < min_fill:
            rejected += 1
            continue

        comp = labels == i
        comp_resp = response[comp]
        if comp_resp.size == 0:
            rejected += 1
            continue

        if float(np.max(comp_resp)) < min_peak and float(np.mean(comp_resp)) < min_mean:
            rejected += 1
            continue

        out[comp] = 1
        kept += 1

    # Slight halo expansion after component filtering.
    dilate = int(p.get("dust_dilate", 2))
    if dilate > 0 and np.any(out):
        k = dilate * 2 + 1
        out = cv2.dilate(
            out,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )

    out = (out.astype(bool) & valid & ~scratch_mask).astype(np.uint8)

    max_fraction = float(p.get("dust_max_mask_fraction", 0.0015))
    frac = float(np.mean(out > 0))
    if frac > max_fraction:
        # Keep strongest components until budget is reached.
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        comps = []
        for i in range(1, num2):
            comp = labels2 == i
            score = float(np.max(response[comp])) + 0.25 * float(np.mean(response[comp]))
            area = int(stats2[i, cv2.CC_STAT_AREA])
            comps.append((score, area, i))
        comps.sort(reverse=True)

        limited = np.zeros_like(out, dtype=np.uint8)
        limit_px = int(max_fraction * frame_area)
        used = 0
        for _, area, i in comps:
            if used + area > limit_px and used > 0:
                continue
            comp = labels2 == i
            limited[comp] = 1
            used += area
            if used >= limit_px:
                break
        out = limited
        print(f"[DUST SAFETY] mask capped: {frac * 100:.3f}% -> {np.mean(out > 0) * 100:.3f}%")

    if bool(p.get("debug_dust", True)):
        print(
            f"[DUST DETECT] kept={kept}, rejected={rejected}, "
            f"mask_px={int(np.sum(out > 0))}, mask_fraction={np.mean(out > 0) * 100:.4f}%"
        )

    return out.astype(bool)


def detect_dust(
    aligned_others,
    center=None,
    p=None,
    valid_mask: Optional[np.ndarray] = None,
    scratch_mask: Optional[np.ndarray] = None,
):
    """
    Compatible dust API.

    Supported calls:
        detect_dust(others, center, p, valid_mask=..., scratch_mask=...)
        detect_dust(others, center, threshold_float)  # older restore.py style

    Returns:
        (dust_mask, large_dust_mask)
    """
    # Backward compatibility with old positional usage.
    if center is None:
        raise ValueError("detect_dust requires center frame")

    if p is None:
        p = {}
    if not isinstance(p, dict):
        p = {"dust_abs": float(p), "dust": float(p)}

    gray = _to_gray(center)
    h, w = gray.shape

    valid = _as_bool_mask(valid_mask, (h, w), default=True)
    scratch = _as_bool_mask(scratch_mask, (h, w), default=False)

    # Build local contrast.
    bg_ksize = _odd(p.get("dust_bg_size", 31), 9)
    bg = cv2.GaussianBlur(gray, (bg_ksize, bg_ksize), 0)
    noise = _local_noise(gray, bg, int(p.get("dust_noise_box", 61)))

    bright_resp = gray - bg
    dark_resp = bg - gray

    dust_abs = float(p.get("dust_abs", p.get("dust", 0.045)))
    noise_mul = float(p.get("dust_noise_mul", 3.0))

    temporal_gate = np.ones((h, w), dtype=bool)
    if bool(p.get("dust_use_temporal", True)):
        other_grays = []
        for o in aligned_others or []:
            if o is None:
                continue
            og = _to_gray(o)
            if og.shape == gray.shape:
                other_grays.append(og)
        temporal_gate = _temporal_outlier(gray, other_grays, p)

    response = np.zeros((h, w), dtype=np.float32)
    candidate = np.zeros((h, w), dtype=bool)

    if bool(p.get("dust_detect_bright", True)):
        br = bright_resp > np.maximum(dust_abs, noise_mul * noise)
        response = np.maximum(response, bright_resp)
        candidate |= br

    if bool(p.get("dust_detect_dark", False)):
        dr = dark_resp > np.maximum(float(p.get("dust_dark_abs", dust_abs)), noise_mul * noise)
        response = np.maximum(response, dark_resp)
        candidate |= dr

    candidate &= temporal_gate
    candidate &= valid
    candidate &= ~scratch

    dust = _component_filter(candidate, response, valid, scratch, p)

    # Keep API compatible with older restore.py: second mask reserved for larger blobs.
    large_dust = np.zeros_like(dust, dtype=bool)
    return dust.astype(bool), large_dust
