"""
scratch_ac_v12.py

Scratch detection only.

This version replaces the earlier component/budget-based scratch path with a
locally adaptive line-segment detector inspired by Newson/Almansa/Gousseau/Perez:

1. bright-only horizontal profile test for negative-film scans;
2. local detection-density estimation;
3. near-vertical line-segment grouping with an a-contrario-style score;
4. local temporal validation over the current frame window;
5. separate protected masks for thin/sharp and soft/wide scratches.

It returns a boolean mask for the center frame only.  Repair is handled by
restore.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _odd(v: int | float, minimum: int = 3) -> int:
    v = int(round(v))
    v = max(minimum, v)
    if v % 2 == 0:
        v += 1
    return v


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return np.mean(img.astype(np.float32), axis=2)


def _safe_bool_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != shape:
        m = cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _build_valid_mask(shape: Tuple[int, int], p: Dict, preset_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Build the valid processing region.

    v9 change:
    The old mask-boundary guard used repeated rectangular erosion.  That helped,
    but could behave unpredictably around irregular preset-mask/film-gate edges.
    This version uses a distance transform so pixels are accepted only if they
    are a known distance away from the valid-mask boundary.
    """
    h, w = shape
    valid = _safe_bool_mask(preset_mask, shape).astype(bool)

    # Optional traditional erosion for overall conservative masking.
    erode_px = int(p.get("mask_edge_erode", 12))
    if erode_px > 0 and np.any(valid):
        k = erode_px * 2 + 1
        valid = cv2.erode(
            valid.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        ).astype(bool)

    # Distance-transform safety from the current valid-mask boundary.
    # This is better than more rectangular erosion because it follows the actual
    # imported mask / film-gate contour.
    boundary_px = int(p.get("mask_boundary_safety_px", 0))
    if boundary_px > 0 and np.any(valid):
        dist = cv2.distanceTransform(
            valid.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        valid = valid & (dist > float(boundary_px))

    border = int(p.get("ignore_frame_border_px", 32))
    if border > 0:
        valid[:, :border] = False
        valid[:, max(0, w - border):] = False
        valid[:border, :] = False
        valid[max(0, h - border):, :] = False

    return valid

def _shift_x(arr: np.ndarray, offset: int) -> np.ndarray:
    """Reflect-padded horizontal shift. Positive offset samples to the right."""
    h, w = arr.shape
    pad = abs(int(offset)) + 8
    padded = cv2.copyMakeBorder(arr, 0, 0, pad, pad, cv2.BORDER_REFLECT_101)
    start = pad + int(offset)
    return padded[:, start:start + w]


def _kl_bernoulli(r: float, q: float) -> float:
    """KL divergence D(Ber(r) || Ber(q)), guarded for 0/1 cases."""
    eps = 1e-6
    r = float(np.clip(r, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return r * math.log(r / q) + (1.0 - r) * math.log((1.0 - r) / (1.0 - q))


def _line_points(x1: float, y1: float, x2: float, y2: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    n = max(n, 2)
    xs = np.round(np.linspace(x1, x2, n)).astype(np.int32)
    ys = np.round(np.linspace(y1, y2, n)).astype(np.int32)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return xs[ok], ys[ok]


def _corridor_values(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, half_width: int) -> np.ndarray:
    h, w = arr.shape
    if xs.size == 0:
        return np.array([], dtype=arr.dtype)
    vals = []
    for dx in range(-half_width, half_width + 1):
        xx = np.clip(xs + dx, 0, w - 1)
        vals.append(arr[ys, xx])
    return np.stack(vals, axis=1)


def _line_x_at_y(line: Dict, y: float) -> float:
    y1 = float(line["y1"])
    y2 = float(line["y2"])
    x1 = float(line["x1"])
    x2 = float(line["x2"])
    if abs(y2 - y1) < 1e-3:
        return 0.5 * (x1 + x2)
    t = (float(y) - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _y_overlap(a: Dict, b: Dict) -> float:
    ay0, ay1 = sorted((float(a["y1"]), float(a["y2"])))
    by0, by1 = sorted((float(b["y1"]), float(b["y2"])))
    inter = max(0.0, min(ay1, by1) - max(ay0, by0))
    denom = max(1.0, min(ay1 - ay0, by1 - by0))
    return inter / denom


def _same_lane(a: Dict, b: Dict, p: Dict, frame_distance: int = 1) -> bool:
    if _y_overlap(a, b) < float(p.get("temporal_min_y_overlap", 0.20)):
        return False

    ay0, ay1 = sorted((float(a["y1"]), float(a["y2"])))
    by0, by1 = sorted((float(b["y1"]), float(b["y2"])))
    y0 = max(ay0, by0)
    y1 = min(ay1, by1)
    if y1 <= y0:
        return False
    ym = 0.5 * (y0 + y1)
    dx = abs(_line_x_at_y(a, ym) - _line_x_at_y(b, ym))
    max_dx = float(p.get("temporal_max_dx", 24.0)) * max(1, frame_distance)
    if dx > max_dx:
        return False

    if abs(float(a.get("slope", 0.0)) - float(b.get("slope", 0.0))) > float(p.get("temporal_max_slope_delta", 0.75)):
        return False

    return True


# -----------------------------------------------------------------------------
# Pixel-wise horizontal profile tests
# -----------------------------------------------------------------------------


def _profile_response(gray: np.ndarray, valid: np.ndarray, p: Dict, kind: str) -> Dict[str, np.ndarray | float]:
    """
    Bright scratch profile test.

    For negative-film scans, scratches are bright/white.  We compare a central
    vertical stripe with left/right horizontal background samples and require
    left/right coherence to reject ordinary image edges.
    """
    gray_s = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)

    if kind == "thin":
        widths = tuple(int(v) for v in p.get("thin_widths", (1, 3, 5)))
        side_width = _odd(p.get("thin_side_width", 5), 3)
        side_gap = int(p.get("thin_side_gap", 2))
        abs_thr = float(p.get("thin_abs", 0.0070))
        rel_thr = float(p.get("thin_rel", 0.11))
        noise_mul = float(p.get("thin_noise_mul", 1.65))
        side_coh = float(p.get("thin_side_coherence", 0.018))
        side_coh_rel = float(p.get("thin_side_coherence_rel", 0.30))
        side_noise_mul = float(p.get("thin_side_noise_mul", 1.65))
    else:
        widths = tuple(int(v) for v in p.get("soft_widths", (7, 11, 15, 21)))
        side_width = _odd(p.get("soft_side_width", 11), 5)
        side_gap = int(p.get("soft_side_gap", 4))
        abs_thr = float(p.get("soft_abs", 0.0042))
        rel_thr = float(p.get("soft_rel", 0.055))
        noise_mul = float(p.get("soft_noise_mul", 1.10))
        side_coh = float(p.get("soft_side_coherence", 0.030))
        side_coh_rel = float(p.get("soft_side_coherence_rel", 0.55))
        side_noise_mul = float(p.get("soft_side_noise_mul", 2.50))

    side_mean = cv2.blur(gray_s, (side_width, 1))

    best_response = np.zeros_like(gray_s, dtype=np.float32)
    best_bg = np.maximum(gray_s, 1e-6).astype(np.float32)
    best_side_diff = np.full_like(gray_s, np.inf, dtype=np.float32)
    best_width = np.ones_like(gray_s, dtype=np.float32)

    for ww in widths:
        ww = _odd(ww, 1)
        center = gray_s if ww <= 1 else cv2.blur(gray_s, (ww, 1))
        shift = side_gap + ww // 2 + side_width // 2
        left = _shift_x(side_mean, -shift)
        right = _shift_x(side_mean, shift)

        # A true bright scratch should be a ridge, not a one-sided step.
        bg_avg = 0.5 * (left + right)
        bg_max = np.maximum(left, right)
        response_avg = np.maximum(center - bg_avg, 0.0)
        response_max = np.maximum(center - bg_max, 0.0)

        # Soft scratches are often broad and low contrast; use avg response but
        # penalize side incoherence later.  Thin scratches use max response.
        response = response_max if kind == "thin" else response_avg
        side_diff = np.abs(left - right)

        better = response > best_response
        best_response[better] = response[better]
        best_bg[better] = bg_avg[better]
        best_side_diff[better] = side_diff[better]
        best_width[better] = float(ww)

    # Local grain/noise estimator based on high-pass horizontal residual.
    noise_bg_w = _odd(p.get("profile_noise_bg_width", 31), 5)
    noise_box = _odd(p.get("profile_local_noise_box", 51), 9)
    hp = np.abs(gray_s - cv2.blur(gray_s, (noise_bg_w, 1)))
    local_noise = cv2.blur(hp, (noise_box, noise_box))

    rel_floor = float(p.get("scratch_rel_floor", 0.025))
    rel = best_response / np.maximum(best_bg, rel_floor)

    adaptive_abs = np.maximum(abs_thr, local_noise * noise_mul)
    side_thr = side_coh + side_coh_rel * best_response + side_noise_mul * local_noise
    coherent = best_side_diff <= side_thr

    # Local ridge test.  This blocks broad image highlights.
    peak_w = _odd(p.get("thin_peak_width" if kind == "thin" else "soft_peak_width", 9 if kind == "thin" else 21), 3)
    local_max = cv2.dilate(best_response, cv2.getStructuringElement(cv2.MORPH_RECT, (peak_w, 1)), iterations=1)
    ridge = best_response >= (local_max - 1e-6)

    seed = (best_response >= adaptive_abs) & (rel >= rel_thr) & coherent & ridge & valid

    return {
        "response": best_response,
        "rel": rel,
        "local_noise": local_noise,
        "width_map": best_width,
        "side_diff": best_side_diff,
        "coherent": coherent,
        "seed": seed,
        "abs_thr": abs_thr,
        "rel_thr": rel_thr,
    }


def _prune_seed(seed: np.ndarray, response: np.ndarray, p: Dict, kind: str) -> np.ndarray:
    """Bound candidate count before Hough. Keeps the strongest response pixels."""
    max_frac = float(p.get("thin_max_seed_fraction" if kind == "thin" else "soft_max_seed_fraction", 0.020 if kind == "thin" else 0.018))
    max_px = int(p.get("thin_max_seed_pixels" if kind == "thin" else "soft_max_seed_pixels", 260000 if kind == "thin" else 220000))
    h, w = seed.shape
    limit = min(max_px, int(max_frac * h * w))
    count = int(np.sum(seed))
    if count <= limit or limit <= 0:
        return seed

    vals = response[seed]
    if vals.size <= limit:
        return seed
    kth = vals.size - limit
    thr = float(np.partition(vals, kth)[kth])
    out = seed & (response >= thr)
    print(f"[AC PRUNE {kind}] seed_px={count} -> {int(np.sum(out))} limit={limit}")
    return out


# -----------------------------------------------------------------------------
# A-contrario-like line grouping
# -----------------------------------------------------------------------------


def _hough_candidates(seed: np.ndarray, p: Dict, kind: str) -> List[Tuple[int, int, int, int]]:
    if not np.any(seed):
        return []

    if kind == "thin":
        close_h = _odd(p.get("thin_connect_height", 25), 3)
        close_w = _odd(p.get("thin_connect_width", 3), 1)
        min_len = int(p.get("thin_min_line_length", 55))
        max_gap = int(p.get("thin_max_line_gap", 24))
        threshold = int(p.get("thin_hough_threshold", 28))
    else:
        close_h = _odd(p.get("soft_connect_height", 85), 3)
        close_w = _odd(p.get("soft_connect_width", 5), 1)
        min_len = int(p.get("soft_min_line_length", 110))
        max_gap = int(p.get("soft_max_line_gap", 150))
        threshold = int(p.get("soft_hough_threshold", 38))

    closed = cv2.morphologyEx(
        seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        iterations=1,
    )

    lines = cv2.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    out: List[Tuple[int, int, int, int]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dy = y2 - y1
        dx = x2 - x1
        if abs(dy) < 2:
            continue
        slope = dx / float(dy)
        if abs(slope) > float(p.get("thin_max_abs_slope" if kind == "thin" else "soft_max_abs_slope", 0.30 if kind == "thin" else 0.35)):
            continue
        # Normalize top-to-bottom.
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        out.append((x1, y1, x2, y2))

    max_lines = int(p.get("thin_max_hough_lines" if kind == "thin" else "soft_max_hough_lines", 900 if kind == "thin" else 500))
    return out[:max_lines]


def _local_density(seed: np.ndarray, p: Dict, kind: str) -> np.ndarray:
    box = _odd(p.get("density_box", 121), 31)
    density = cv2.blur(seed.astype(np.float32), (box, box))
    return density


def _evaluate_line(raw_line: Tuple[int, int, int, int], prof: Dict, seed: np.ndarray, density: np.ndarray, valid: np.ndarray, p: Dict, kind: str) -> Optional[Dict]:
    x1, y1, x2, y2 = raw_line
    h, w = seed.shape
    xs, ys = _line_points(x1, y1, x2, y2, (h, w))
    if xs.size < 2:
        return None

    length = float(xs.size)
    slope = (x2 - x1) / max(1.0, float(y2 - y1))

    if kind == "thin":
        half = int(p.get("thin_eval_half_width", 1))
        min_len = float(p.get("thin_min_length", 55))
        min_coverage = float(p.get("thin_min_coverage", 0.10))
        min_excess = float(p.get("thin_min_density_excess", 0.030))
        min_log_nfa = float(p.get("thin_min_log_nfa", 14.0))
        min_response = float(p.get("thin_min_mean_response", 0.0045))
        repair_width = int(p.get("thin_repair_width", 3))
    else:
        half = int(p.get("soft_eval_half_width", 4))
        min_len = float(p.get("soft_min_length", 110))
        min_coverage = float(p.get("soft_min_coverage", 0.055))
        min_excess = float(p.get("soft_min_density_excess", 0.020))
        min_log_nfa = float(p.get("soft_min_log_nfa", 18.0))
        min_response = float(p.get("soft_min_mean_response", 0.0025))
        repair_width = int(p.get("soft_repair_width", 13))

    if length < min_len:
        return None

    valid_vals = _corridor_values(valid.astype(np.uint8), xs, ys, half)
    if valid_vals.size == 0 or np.mean(valid_vals) < 0.90:
        return None

    seed_vals = _corridor_values(seed.astype(np.uint8), xs, ys, half)
    resp_vals = _corridor_values(prof["response"], xs, ys, half)
    density_vals = _corridor_values(density, xs, ys, half)

    # A line position is active if any pixel in the corridor is a seed.
    active_per_row = np.max(seed_vals, axis=1) > 0
    active_idx = np.where(active_per_row)[0]

    # IMPORTANT:
    # The Hough line is only a hypothesis.  For each active row, use the
    # strongest response inside the corridor as the supported scratch location.
    # This prevents a wrong diagonal Hough line from becoming a huge repair band.
    if active_idx.size < 2:
        return None

    dx_offsets = np.arange(-half, half + 1, dtype=np.int32)
    best_dx_idx = np.argmax(resp_vals, axis=1)
    supported_x = np.clip(xs + dx_offsets[best_dx_idx], 0, w - 1).astype(np.int32)
    supported_y = ys.astype(np.int32)

    active_x = supported_x[active_idx].astype(np.float32)
    active_y = supported_y[active_idx].astype(np.float32)

    # Reject lines whose actual support wanders too far horizontally.
    # This is stronger than a simple slope test and kills large diagonal
    # false positives while still allowing mild drift.
    x_med = float(np.median(active_x))
    x_mad = float(np.median(np.abs(active_x - x_med)))
    x_span = float(np.percentile(active_x, 95) - np.percentile(active_x, 5))

    if kind == "thin":
        if x_mad > float(p.get("thin_max_x_mad", 2.0)):
            return None
        if x_span > float(p.get("thin_max_x_span", 10.0)):
            return None
    else:
        if x_mad > float(p.get("soft_max_x_mad", 7.0)):
            return None
        if x_span > float(p.get("soft_max_x_span", 28.0)):
            return None

    k = int(active_idx.size)
    r = k / max(1.0, length)
    local_p = float(np.mean(density_vals))
    local_p = max(local_p, 1e-5)

    # A-contrario-style significance score.  Not a full NFA implementation, but
    # it is the crucial missing idea: require a segment to be unlikely relative
    # to the LOCAL clutter/detection density.
    log_nfa = length * _kl_bernoulli(r, min(local_p, 0.95))

    mean_response = float(np.mean(np.max(resp_vals, axis=1)))
    max_response = float(np.max(resp_vals))

    if r < max(min_coverage, local_p + min_excess):
        return None
    if log_nfa < min_log_nfa:
        return None
    if mean_response < min_response and max_response < min_response * 2.5:
        return None

    # Estimate repair width from the detected profile width, but keep hard
    # scratches narrow.  The GUI Max Repair Width caps the soft/wide path.
    width_vals = _corridor_values(prof["width_map"], xs, ys, half)
    active_widths = np.max(width_vals, axis=1)[active_per_row]
    if active_widths.size > 0:
        local_width = float(np.median(active_widths))
    else:
        local_width = float(repair_width)

    if kind == "thin":
        final_repair_width = int(np.clip(
            int(p.get("thin_repair_width", 3)),
            1,
            int(p.get("thin_max_repair_width", 7)),
        ))
        support_x_out = supported_x[active_idx].astype(np.int32)
        support_y_out = supported_y[active_idx].astype(np.int32)
        fill_added_rows = 0
    else:
        # Soft/wide scratches were being repaired as 5–6 px fragments.
        # Keep the line evidence strict, but widen accepted soft lines enough to
        # cover the visible halo.
        soft_min_w = int(p.get("soft_line_min_repair_width", 13))
        soft_max_w = max(soft_min_w, int(p.get("soft_max_repair_width", 25)))
        final_repair_width = int(np.clip(
            round(local_width + float(p.get("soft_repair_extra", 2.0))),
            soft_min_w,
            soft_max_w,
        ))
        if final_repair_width % 2 == 0:
            final_repair_width += 1

        support_x_out = supported_x[active_idx].astype(np.int32)
        support_y_out = supported_y[active_idx].astype(np.int32)
        fill_added_rows = 0

        # v7 soft-line fill:
        # Once a soft line has passed the AC/temporal tests, recover weaker rows
        # between nearby strong anchor rows along that same line.  This reduces
        # the dotted/gapped mask without drawing full-height columns.
        if bool(p.get("enable_soft_line_fill", True)):
            strong_rows = supported_y[active_idx].astype(np.int32)
            strong_set = set(int(v) for v in strong_rows.tolist())
            fill_anchor_gap = int(p.get("soft_line_fill_max_anchor_gap", 420))
            fill_abs = float(p.get("soft_line_fill_abs", 0.00035))
            fill_rel = float(p.get("soft_line_fill_rel_factor", 0.14))
            fill_solid_abs = float(p.get("soft_line_fill_solid_abs", 0.00018))
            fill_solid_gap = int(p.get("soft_line_fill_solid_max_gap", 260))

            filled_indices = []
            if strong_rows.size >= 2:
                for ii in range(len(ys)):
                    yy_abs = int(supported_y[ii])
                    prev_rows = strong_rows[strong_rows <= yy_abs]
                    next_rows = strong_rows[strong_rows >= yy_abs]
                    if not (prev_rows.size and next_rows.size):
                        continue

                    gap = int(next_rows[0] - prev_rows[-1])
                    between_anchors = gap <= fill_anchor_gap
                    row_best_resp = float(np.max(resp_vals[ii]))

                    weak_thr = max(fill_abs, mean_response * fill_rel)
                    solid_thr = min(weak_thr, fill_solid_abs)

                    if yy_abs in strong_set:
                        filled_indices.append(ii)
                    elif between_anchors and row_best_resp >= weak_thr:
                        filled_indices.append(ii)
                    elif gap <= fill_solid_gap and row_best_resp >= solid_thr:
                        filled_indices.append(ii)

            if len(filled_indices) >= max(2, int(p.get("soft_line_fill_min_rows", 8))):
                filled_indices = np.asarray(sorted(set(filled_indices)), dtype=np.int32)
                support_x_out = supported_x[filled_indices].astype(np.int32)
                support_y_out = supported_y[filled_indices].astype(np.int32)
                fill_added_rows = max(0, int(len(filled_indices) - len(active_idx)))

    score = log_nfa + 200.0 * mean_response + 0.015 * length + 0.003 * fill_added_rows

    return {
        "kind": kind,
        "source": "soft_line" if kind == "soft" else "thin_line",
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "length": length,
        "slope": float(slope),
        "coverage": float(r),
        "fill_added_rows": int(fill_added_rows),
        "local_density": float(local_p),
        "log_nfa": float(log_nfa),
        "mean_response": mean_response,
        "max_response": max_response,
        "score": float(score),
        "repair_width": int(max(1, final_repair_width)),
        "votes": 1,
        # Supported coordinates are the actual geometry used for repair.
        # The line endpoints remain only for temporal matching and display/debug.
        "support_x": support_x_out,
        "support_y": support_y_out,
    }


def _suppress_duplicate_lines(lines: List[Dict], p: Dict, kind: str) -> List[Dict]:
    if not lines:
        return []
    lines = sorted(lines, key=lambda d: d["score"], reverse=True)
    kept: List[Dict] = []
    max_keep = int(p.get("thin_max_lines" if kind == "thin" else "soft_max_lines", 120 if kind == "thin" else 24))
    x_tol = float(p.get("thin_nms_x" if kind == "thin" else "soft_nms_x", 10 if kind == "thin" else 24))
    y_ov = float(p.get("line_nms_y_overlap", 0.35))
    for line in lines:
        duplicate = False
        for k in kept:
            if _y_overlap(line, k) >= y_ov:
                ym = 0.5 * (max(min(line["y1"], line["y2"]), min(k["y1"], k["y2"])) + min(max(line["y1"], line["y2"]), max(k["y1"], k["y2"])))
                if abs(_line_x_at_y(line, ym) - _line_x_at_y(k, ym)) <= x_tol:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(line)
            if len(kept) >= max_keep:
                break
    return kept



def _irregular_emulsion_damage_recovery(prof: Dict, valid: np.ndarray, p: Dict) -> List[Dict]:
    """
    Recover thicker, irregular bright scratch/emulsion-damage fragments.

    The normal thin/soft paths are line/ridge based.  They work well for sharp
    1-3 px scratches, but they intentionally reject irregular leftover emulsion
    patches because those patches are not clean straight ridges.  This detector
    is component/track based:

    - uses a relaxed bright-horizontal-profile response, without the strict
      local-ridge requirement;
    - keeps only vertically biased components;
    - represents each component as row-wise support_x/support_y, so repair still
      uses supported geometry rather than full columns;
    - leaves long-gap bridging to the soft-track fill stage.
    """
    if not bool(p.get("enable_irregular_emulsion_damage", True)):
        return []

    response = np.asarray(prof["response"], dtype=np.float32)
    rel = np.asarray(prof["rel"], dtype=np.float32)
    local_noise = np.asarray(prof["local_noise"], dtype=np.float32)
    side_diff = np.asarray(prof["side_diff"], dtype=np.float32)
    width_map = np.asarray(prof["width_map"], dtype=np.float32)

    h, w = response.shape

    abs_thr = float(p.get("emulsion_abs", 0.0026))
    rel_thr = float(p.get("emulsion_rel", 0.020))
    noise_mul = float(p.get("emulsion_noise_mul", 0.60))

    # Relaxed side coherence.  This is intentionally more permissive than the
    # soft ridge path because irregular emulsion damage may have uneven edges.
    side_thr = (
        float(p.get("emulsion_side_coherence", 0.055))
        + float(p.get("emulsion_side_coherence_rel", 0.90)) * response
        + float(p.get("emulsion_side_noise_mul", 3.50)) * local_noise
    )

    candidate = (
        (response >= np.maximum(abs_thr, local_noise * noise_mul)) &
        (rel >= rel_thr) &
        (side_diff <= side_thr) &
        valid
    )

    # Remove very isolated grain, then connect vertically biased fragments.
    open_w = int(p.get("emulsion_open_width", 1))
    open_h = int(p.get("emulsion_open_height", 1))
    if open_w > 1 or open_h > 1:
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_w, 1), _odd(open_h, 1))),
            iterations=1,
        ).astype(bool)

    close_w = _odd(p.get("emulsion_connect_width", 9), 1)
    close_h = _odd(p.get("emulsion_connect_height", 71), 3)
    connected = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        iterations=1,
    )

    dil_w = int(p.get("emulsion_dilate_width", 3))
    if dil_w > 1:
        connected = cv2.dilate(
            connected,
            cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(dil_w, 1), 1)),
            iterations=1,
        )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)

    out: List[Dict] = []

    min_area = int(p.get("emulsion_min_area", 28))
    max_area = int(p.get("emulsion_max_area", 16000))
    min_height = int(p.get("emulsion_min_height", 35))
    max_width = int(p.get("emulsion_max_width", 82))
    min_aspect = float(p.get("emulsion_min_aspect", 1.35))
    min_active_rows = int(p.get("emulsion_min_active_rows", 14))
    min_active_fraction = float(p.get("emulsion_min_active_fraction", 0.035))
    min_mean_response = float(p.get("emulsion_min_mean_response", 0.00075))
    min_peak_response = float(p.get("emulsion_min_peak_response", 0.0020))
    max_x_mad = float(p.get("emulsion_max_x_mad", 16.0))
    max_x_span = float(p.get("emulsion_max_x_span", 88.0))
    max_slope = float(p.get("emulsion_max_abs_slope", p.get("soft_max_abs_slope", 0.18)))
    max_lines = int(p.get("emulsion_max_lines", 48))

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
        if ch < min_height or cw > max_width:
            rejected += 1
            continue
        if ch / max(1.0, float(cw)) < min_aspect:
            rejected += 1
            continue

        comp = labels == i

        # Use original candidate pixels inside the connected component for
        # support, not the whole closed rectangle.
        support_region = comp & candidate
        ys_all, xs_all = np.where(support_region)
        if ys_all.size < min_active_rows:
            rejected += 1
            continue

        unique_rows = np.unique(ys_all)
        if unique_rows.size < min_active_rows:
            rejected += 1
            continue
        if unique_rows.size / max(1.0, float(ch)) < min_active_fraction:
            rejected += 1
            continue

        xs_support = []
        ys_support = []
        resp_support = []
        width_support = []

        for yy in unique_rows:
            row_xs = xs_all[ys_all == yy]
            if row_xs.size == 0:
                continue
            vals = response[yy, row_xs]
            best_idx = int(np.argmax(vals))
            best_x = int(row_xs[best_idx])
            xs_support.append(best_x)
            ys_support.append(int(yy))
            resp_support.append(float(vals[best_idx]))
            width_support.append(float(width_map[yy, best_x]))

        if len(xs_support) < min_active_rows:
            rejected += 1
            continue

        xs_arr = np.asarray(xs_support, dtype=np.float32)
        ys_arr = np.asarray(ys_support, dtype=np.float32)
        resp_arr = np.asarray(resp_support, dtype=np.float32)
        width_arr = np.asarray(width_support, dtype=np.float32)

        mean_response = float(np.mean(resp_arr))
        peak_response = float(np.max(resp_arr))
        if mean_response < min_mean_response and peak_response < min_peak_response:
            rejected += 1
            continue

        x_med = float(np.median(xs_arr))
        x_mad = float(np.median(np.abs(xs_arr - x_med)))
        x_span = float(np.percentile(xs_arr, 95) - np.percentile(xs_arr, 5))
        if x_mad > max_x_mad:
            rejected += 1
            continue
        if x_span > max_x_span:
            rejected += 1
            continue

        if np.ptp(ys_arr) > 1:
            slope, intercept = np.polyfit(ys_arr, xs_arr, 1)
        else:
            slope = 0.0
            intercept = x_med

        if abs(float(slope)) > max_slope:
            rejected += 1
            continue

        y_top = float(np.min(ys_arr))
        y_bot = float(np.max(ys_arr))
        x_top = float(slope * y_top + intercept)
        x_bot = float(slope * y_bot + intercept)
        span = max(1.0, y_bot - y_top + 1.0)

        # Use a broad practical width for irregular emulsion fragments.
        repair_width = int(p.get("emulsion_repair_width", p.get("soft_max_repair_width", 31)))
        repair_width = int(np.clip(
            repair_width,
            int(p.get("emulsion_min_repair_width", 23)),
            max(int(p.get("emulsion_min_repair_width", 23)), int(p.get("soft_max_repair_width", 31))),
        ))
        if repair_width % 2 == 0:
            repair_width += 1

        fill = float(area) / max(1.0, float(cw * ch))
        score = (
            16.0
            + 0.10 * span
            + 220.0 * mean_response
            + 20.0 * min(1.0, unique_rows.size / span)
            - 0.20 * x_mad
            - 3.0 * max(0.0, fill - 0.45)
        )

        out.append({
            "kind": "soft",
            "source": "emulsion_damage",
            "x1": float(x_top),
            "y1": float(y_top),
            "x2": float(x_bot),
            "y2": float(y_bot),
            "length": float(span),
            "slope": float(slope),
            "coverage": float(unique_rows.size / span),
            "fill_added_rows": 0,
            "local_density": 0.0,
            "log_nfa": float(score),
            "mean_response": mean_response,
            "max_response": peak_response,
            "score": float(score),
            "repair_width": int(max(1, repair_width)),
            "votes": 1,
            "support_x": np.asarray(xs_support, dtype=np.int32),
            "support_y": np.asarray(ys_support, dtype=np.int32),
        })
        kept += 1

    out.sort(key=lambda d: d["score"], reverse=True)
    if bool(p.get("debug_emulsion_damage", True)):
        print(f"[EMULSION DAMAGE] kept={kept}, rejected={rejected}")

    return out[:max_lines]



def _emulsion_vertical_track_recovery(prof: Dict, valid: np.ndarray, p: Dict) -> List[Dict]:
    """
    Projection-based recovery for irregular emulsion/scratch remnants.

    The component version can merge relaxed candidates into one big component and
    reject it.  This version looks for repeated bright irregular evidence inside
    the same narrow x-lane, then builds a row-wise repair track from that lane.
    It is meant for the "leftover scratched emulsion layer" pieces that are
    not clean ridges but still align vertically with the scratch.
    """
    if not bool(p.get("enable_emulsion_vertical_track", True)):
        return []

    response = np.asarray(prof["response"], dtype=np.float32)
    rel = np.asarray(prof["rel"], dtype=np.float32)
    local_noise = np.asarray(prof["local_noise"], dtype=np.float32)
    side_diff = np.asarray(prof["side_diff"], dtype=np.float32)

    h, w = response.shape

    abs_thr = float(p.get("emulsion_track_abs", p.get("emulsion_abs", 0.0022)))
    rel_thr = float(p.get("emulsion_track_rel", p.get("emulsion_rel", 0.014)))
    noise_mul = float(p.get("emulsion_track_noise_mul", 0.42))
    side_thr = (
        float(p.get("emulsion_track_side_coherence", 0.075))
        + float(p.get("emulsion_track_side_coherence_rel", 1.10)) * response
        + float(p.get("emulsion_track_side_noise_mul", 4.50)) * local_noise
    )

    # Candidate used only for lane discovery.  It is permissive but still based
    # on horizontal scratch excess, not raw brightness.
    cand = (
        (response >= np.maximum(abs_thr, local_noise * noise_mul)) &
        (rel >= rel_thr) &
        (side_diff <= side_thr) &
        valid
    )

    # Suppress isolated grain before vertical projection.
    open_h = int(p.get("emulsion_track_open_height", 3))
    if open_h > 1:
        cand = cv2.morphologyEx(
            cand.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, _odd(open_h, 1))),
            iterations=1,
        ).astype(bool)

    # Column projection.  Smooth in x so irregular wider remnants become one lane.
    proj_x_width = _odd(p.get("emulsion_track_projection_width", 17), 3)
    col_count = np.sum(cand.astype(np.float32), axis=0)
    col_resp = np.sum((response * cand.astype(np.float32)), axis=0)
    col_count_s = cv2.blur(col_count.reshape(1, -1), (proj_x_width, 1)).reshape(-1)
    col_resp_s = cv2.blur(col_resp.reshape(1, -1), (proj_x_width, 1)).reshape(-1)

    min_col_rows = float(p.get("emulsion_track_min_col_rows", 28))
    min_col_response = float(p.get("emulsion_track_min_col_response", 0.030))

    x_seed = (col_count_s >= min_col_rows) & (col_resp_s >= min_col_response)

    # Avoid selecting the outer film-gate/mask boundary as a lane.
    x_seed[: int(p.get("emulsion_track_ignore_border_x", 12))] = False
    x_seed[w - int(p.get("emulsion_track_ignore_border_x", 12)) :] = False

    x_runs = []
    in_run = False
    run_start = 0
    for x in range(w):
        if x_seed[x] and not in_run:
            in_run = True
            run_start = x
        elif not x_seed[x] and in_run:
            x_runs.append((run_start, x - 1))
            in_run = False
    if in_run:
        x_runs.append((run_start, w - 1))

    out: List[Dict] = []

    max_lane_width = int(p.get("emulsion_track_max_lane_width", 96))
    min_lane_width = int(p.get("emulsion_track_min_lane_width", 1))
    lane_pad = int(p.get("emulsion_track_lane_pad", 10))
    min_active_rows = int(p.get("emulsion_track_min_active_rows", 18))
    min_span = int(p.get("emulsion_track_min_span", 160))
    min_active_fraction = float(p.get("emulsion_track_min_active_fraction", 0.020))
    max_gap = int(p.get("emulsion_track_bridge_gap", 900))
    weak_abs = float(p.get("emulsion_track_fill_abs", 0.00028))
    weak_rel_factor = float(p.get("emulsion_track_fill_rel_factor", 0.16))
    max_x_mad = float(p.get("emulsion_track_max_x_mad", 18.0))
    max_x_span = float(p.get("emulsion_track_max_x_span", 96.0))
    max_slope = float(p.get("emulsion_track_max_abs_slope", 0.24))
    max_tracks = int(p.get("emulsion_track_max_tracks", 24))

    kept = 0
    rejected = 0

    for x0, x1 in x_runs:
        lane_w = x1 - x0 + 1
        if lane_w < min_lane_width or lane_w > max_lane_width:
            rejected += 1
            continue

        xl = max(0, x0 - lane_pad)
        xr = min(w, x1 + lane_pad + 1)
        if xr <= xl:
            rejected += 1
            continue

        sub_cand = cand[:, xl:xr]
        sub_resp = response[:, xl:xr]
        sub_valid = valid[:, xl:xr]

        row_has_anchor = np.any(sub_cand & sub_valid, axis=1)
        anchor_rows = np.where(row_has_anchor)[0]
        if anchor_rows.size < min_active_rows:
            rejected += 1
            continue

        y_min = int(anchor_rows.min())
        y_max = int(anchor_rows.max())
        span = y_max - y_min + 1
        if span < min_span:
            rejected += 1
            continue
        if anchor_rows.size / max(1.0, float(span)) < min_active_fraction:
            rejected += 1
            continue

        # Row-wise best anchor positions.
        anchor_xs = []
        anchor_ys = []
        anchor_resp = []
        for yy in anchor_rows:
            valid_cols = np.where((sub_cand[yy] & sub_valid[yy]))[0]
            if valid_cols.size == 0:
                continue
            vals = sub_resp[yy, valid_cols]
            bi = int(valid_cols[int(np.argmax(vals))])
            anchor_xs.append(int(xl + bi))
            anchor_ys.append(int(yy))
            anchor_resp.append(float(sub_resp[yy, bi]))

        if len(anchor_xs) < min_active_rows:
            rejected += 1
            continue

        ax = np.asarray(anchor_xs, dtype=np.float32)
        ay = np.asarray(anchor_ys, dtype=np.float32)

        x_med = float(np.median(ax))
        x_mad = float(np.median(np.abs(ax - x_med)))
        x_span = float(np.percentile(ax, 95) - np.percentile(ax, 5))
        if x_mad > max_x_mad or x_span > max_x_span:
            rejected += 1
            continue

        if np.ptp(ay) > 1:
            slope, intercept = np.polyfit(ay, ax, 1)
        else:
            slope = 0.0
            intercept = x_med
        if abs(float(slope)) > max_slope:
            rejected += 1
            continue

        # Fill between anchors only.  Rows in large anchor gaps require weak
        # response; short gaps can be filled solidly.
        support_x = []
        support_y = []
        row_best_resp = np.max(sub_resp, axis=1)
        row_best_idx = np.argmax(sub_resp, axis=1)

        anchor_rows_sorted = np.asarray(sorted(set(int(v) for v in anchor_ys)), dtype=np.int32)
        mean_anchor_resp = float(np.mean(anchor_resp)) if anchor_resp else 0.0
        weak_thr = max(weak_abs, mean_anchor_resp * weak_rel_factor)
        solid_gap = int(p.get("emulsion_track_solid_gap", 220))

        for yy in range(y_min, y_max + 1):
            prev_rows = anchor_rows_sorted[anchor_rows_sorted <= yy]
            next_rows = anchor_rows_sorted[anchor_rows_sorted >= yy]
            if not (prev_rows.size and next_rows.size):
                continue

            gap = int(next_rows[0] - prev_rows[-1])
            if gap > max_gap:
                continue

            pred_x = int(round(float(slope) * float(yy) + float(intercept)))
            pred_x = int(np.clip(pred_x, xl, xr - 1))

            # Use best local x near predicted path, not necessarily the whole lane.
            half = int(p.get("emulsion_track_fill_half_width", 14))
            sx0 = max(xl, pred_x - half)
            sx1 = min(xr, pred_x + half + 1)
            local = response[yy, sx0:sx1]
            if local.size:
                local_best = int(np.argmax(local))
                best_x = sx0 + local_best
                best_resp = float(local[local_best])
            else:
                best_x = pred_x
                best_resp = float(row_best_resp[yy])

            fill = False
            if yy in anchor_rows_sorted:
                fill = True
            elif gap <= solid_gap:
                fill = best_resp >= max(weak_abs * 0.5, weak_thr * 0.5)
            else:
                fill = best_resp >= weak_thr

            if fill:
                support_x.append(int(best_x))
                support_y.append(int(yy))

        if len(support_y) < min_active_rows:
            rejected += 1
            continue

        repair_width = int(p.get("emulsion_track_repair_width", p.get("emulsion_repair_width", p.get("soft_max_repair_width", 31))))
        repair_width = int(np.clip(
            repair_width,
            int(p.get("emulsion_track_min_repair_width", p.get("emulsion_min_repair_width", 25))),
            max(int(p.get("emulsion_track_min_repair_width", p.get("emulsion_min_repair_width", 25))), int(p.get("soft_max_repair_width", 31))),
        ))
        if repair_width % 2 == 0:
            repair_width += 1

        fill_added = max(0, len(support_y) - len(anchor_rows_sorted))
        score = (
            24.0
            + 0.10 * float(len(support_y))
            + 200.0 * mean_anchor_resp
            + 4.0 * min(1.0, float(len(anchor_rows_sorted)) / float(span))
            - 0.25 * x_mad
        )

        out.append({
            "kind": "soft",
            "source": "emulsion_track",
            "x1": float(slope * float(y_min) + intercept),
            "y1": float(y_min),
            "x2": float(slope * float(y_max) + intercept),
            "y2": float(y_max),
            "length": float(len(support_y)),
            "slope": float(slope),
            "coverage": float(len(anchor_rows_sorted) / max(1.0, float(span))),
            "fill_added_rows": int(fill_added),
            "local_density": 0.0,
            "log_nfa": float(score),
            "mean_response": mean_anchor_resp,
            "max_response": float(max(anchor_resp)) if anchor_resp else 0.0,
            "score": float(score),
            "repair_width": int(max(1, repair_width)),
            "votes": 1,
            "support_x": np.asarray(support_x, dtype=np.int32),
            "support_y": np.asarray(support_y, dtype=np.int32),
        })
        kept += 1

    out.sort(key=lambda d: d["score"], reverse=True)
    if bool(p.get("debug_emulsion_track", True)):
        print(f"[EMULSION TRACK] kept={kept}, rejected={rejected}, x_runs={len(x_runs)}")

    return out[:max_tracks]



def _soft_lane_recovery(prof: Dict, seed: np.ndarray, valid: np.ndarray, p: Dict) -> List[Dict]:
    """
    Recover faint/wide intermittent scratches as near-vertical lanes.

    This is deliberately different from drawing full columns:
    - it starts from the already filtered AC soft seed map;
    - it closes only vertically to connect intermittent evidence;
    - it rejects components with too much horizontal wandering;
    - it stores only supported rows, so the final mask still uses evidence-based
      geometry and bridges only limited gaps.
    """
    if not bool(p.get("enable_soft_lane_recovery", True)):
        return []

    h, w = seed.shape
    response = prof["response"]
    width_map = prof["width_map"]

    close_h = _odd(p.get("soft_lane_connect_height", 151), 3)
    close_w = _odd(p.get("soft_lane_connect_width", 3), 1)

    # Work from the seed, not from raw response, so local profile/coherence tests
    # are still respected.
    lane = cv2.morphologyEx(
        (seed & valid).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        iterations=1,
    )

    # A tiny horizontal dilation helps broad low-contrast scratches become one
    # component, but avoid wide bands.
    dil_w = int(p.get("soft_lane_dilate_width", 3))
    if dil_w > 1:
        lane = cv2.dilate(
            lane,
            cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(dil_w, 1), 1)),
            iterations=1,
        )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(lane.astype(np.uint8), connectivity=8)

    out: List[Dict] = []

    min_span = int(p.get("soft_lane_min_span", 120))
    max_width = int(p.get("soft_lane_max_width", 42))
    min_aspect = float(p.get("soft_lane_min_aspect", 3.5))
    min_active_rows = int(p.get("soft_lane_min_active_rows", 18))
    min_active_fraction = float(p.get("soft_lane_min_active_fraction", 0.035))
    min_mean_response = float(p.get("soft_lane_min_mean_response", 0.0010))
    min_peak_response = float(p.get("soft_lane_min_peak_response", 0.0030))
    max_abs_slope = float(p.get("soft_max_abs_slope", 0.18))
    max_x_mad = float(p.get("soft_max_x_mad", 7.0))
    max_x_span = float(p.get("soft_max_x_span", 28.0))
    max_lines = int(p.get("soft_lane_max_lines", 30))

    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if ch < min_span:
            continue
        if cw > max_width:
            continue
        if ch / max(1.0, float(cw)) < min_aspect:
            continue

        x0 = max(0, x - int(p.get("soft_lane_x_pad", 3)))
        x1 = min(w, x + cw + int(p.get("soft_lane_x_pad", 3)))
        y0 = max(0, y)
        y1 = min(h, y + ch)

        # For each row, accept it only if real seed evidence exists in this lane.
        sub_seed = seed[y0:y1, x0:x1] & valid[y0:y1, x0:x1]
        sub_resp = response[y0:y1, x0:x1]
        sub_width = width_map[y0:y1, x0:x1]

        active = np.any(sub_seed, axis=1)
        active_rows_local = np.where(active)[0]
        if active_rows_local.size < min_active_rows:
            continue
        if active_rows_local.size / max(1.0, float(ch)) < min_active_fraction:
            continue

        xs_support = []
        ys_support = []
        resp_support = []
        width_support = []

        for yy in active_rows_local:
            row_seed = sub_seed[yy]
            if not np.any(row_seed):
                continue
            # Pick the strongest response among the seeded pixels in this row.
            candidate_xs = np.where(row_seed)[0]
            vals = sub_resp[yy, candidate_xs]
            best = int(candidate_xs[int(np.argmax(vals))])
            xs_support.append(x0 + best)
            ys_support.append(y0 + yy)
            resp_support.append(float(sub_resp[yy, best]))
            width_support.append(float(sub_width[yy, best]))

        if len(xs_support) < min_active_rows:
            continue

        xs_arr = np.asarray(xs_support, dtype=np.float32)
        ys_arr = np.asarray(ys_support, dtype=np.float32)
        resp_arr = np.asarray(resp_support, dtype=np.float32)
        width_arr = np.asarray(width_support, dtype=np.float32)

        mean_response = float(np.mean(resp_arr))
        peak_response = float(np.max(resp_arr))
        if mean_response < min_mean_response and peak_response < min_peak_response:
            continue

        x_med = float(np.median(xs_arr))
        x_mad = float(np.median(np.abs(xs_arr - x_med)))
        x_span = float(np.percentile(xs_arr, 95) - np.percentile(xs_arr, 5))
        if x_mad > max_x_mad:
            continue
        if x_span > max_x_span:
            continue

        # Fit a mild line only for temporal matching/debug; repair uses support.
        if ys_arr.size >= 2 and np.ptp(ys_arr) > 1:
            slope, intercept = np.polyfit(ys_arr, xs_arr, 1)
        else:
            slope = 0.0
            intercept = x_med

        if abs(float(slope)) > max_abs_slope:
            continue

        y_top_i = int(np.min(ys_arr))
        y_bot_i = int(np.max(ys_arr))
        y_top = float(y_top_i)
        y_bot = float(y_bot_i)
        x_top = float(slope * y_top + intercept)
        x_bot = float(slope * y_bot + intercept)
        span = max(1.0, y_bot - y_top + 1.0)

        # v6: soft-lane repair width is intentionally controlled by the GUI cap.
        # The previous adaptive width often collapsed to 5-6 px, which was too
        # narrow for diffuse soft scratches.
        local_width = float(np.median(width_arr)) if width_arr.size else float(p.get("soft_repair_width", 13))
        repair_width = int(np.clip(
            int(p.get("soft_lane_repair_width", p.get("soft_max_repair_width", 25))),
            int(p.get("soft_lane_min_repair_width", 9)),
            int(p.get("soft_max_repair_width", 25)),
        ))
        if repair_width % 2 == 0:
            repair_width += 1

        # v6 lane-fill:
        # Once a soft lane is validated, recover faint rows along that lane with
        # a much weaker row test. This is still constrained to the detected lane
        # span and x-position, so it should not create full-frame bands.
        fill_xs = list(xs_support)
        fill_ys = list(ys_support)
        fill_added_rows = 0

        if bool(p.get("enable_soft_lane_fill", True)):
            fill_half_width = int(p.get("soft_lane_fill_half_width", max(3, repair_width // 2)))
            fill_abs = float(p.get("soft_lane_fill_abs", 0.00035))
            fill_rel = float(p.get("soft_lane_fill_rel_factor", 0.12))
            fill_solid_abs = float(p.get("soft_lane_fill_solid_abs", 0.00018))
            fill_anchor_gap = int(p.get("soft_lane_fill_max_anchor_gap", 480))
            solid_gap = int(p.get("soft_lane_fill_solid_max_gap", 480))

            strong_rows = np.array(sorted(set(int(yv) for yv in ys_support)), dtype=np.int32)

            if strong_rows.size >= 2:
                candidate_rows = np.arange(y_top_i, y_bot_i + 1, dtype=np.int32)
                filled = []
                strong_set = set(int(v) for v in strong_rows.tolist())
                for yy_abs in candidate_rows:
                    yy = yy_abs - y0
                    if yy < 0 or yy >= sub_resp.shape[0]:
                        continue

                    x_pred_abs = int(round(float(slope) * float(yy_abs) + float(intercept)))
                    xx0_abs = max(x0, x_pred_abs - fill_half_width)
                    xx1_abs = min(x1, x_pred_abs + fill_half_width + 1)
                    if xx1_abs <= xx0_abs:
                        continue

                    xx0 = xx0_abs - x0
                    xx1 = xx1_abs - x0
                    row_resp = sub_resp[yy, xx0:xx1]
                    if row_resp.size == 0:
                        continue

                    best_local = int(np.argmax(row_resp))
                    best_resp = float(row_resp[best_local])
                    best_x_abs = int(xx0_abs + best_local)

                    prev_rows = strong_rows[strong_rows <= yy_abs]
                    next_rows = strong_rows[strong_rows >= yy_abs]
                    between_anchors = False
                    gap = 999999
                    if prev_rows.size and next_rows.size:
                        gap = int(next_rows[0] - prev_rows[-1])
                        between_anchors = gap <= fill_anchor_gap

                    weak_thr = max(fill_abs, mean_response * fill_rel)
                    solid_thr = min(weak_thr, fill_solid_abs)

                    keep_row = False
                    if yy_abs in strong_set:
                        keep_row = True
                    elif between_anchors and best_resp >= weak_thr:
                        keep_row = True
                    elif gap <= solid_gap and best_resp >= solid_thr:
                        # Very weak fill only between reasonably close anchors.
                        keep_row = True

                    if keep_row:
                        filled.append((best_x_abs, int(yy_abs)))

                if len(filled) >= min_active_rows:
                    fill_xs = [x for x, y in filled]
                    fill_ys = [y for x, y in filled]
                    fill_added_rows = max(0, len(fill_ys) - len(xs_support))

        fill_xs_arr = np.asarray(fill_xs, dtype=np.int32)
        fill_ys_arr = np.asarray(fill_ys, dtype=np.int32)

        coverage = len(xs_support) / span
        fill_coverage = len(fill_ys_arr) / span
        score = (
            18.0
            + 0.08 * span
            + 160.0 * mean_response
            + 8.0 * min(1.0, coverage)
            + 4.0 * min(1.0, fill_coverage)
            - 0.4 * x_mad
        )

        out.append({
            "kind": "soft",
            "source": "soft_lane",
            "x1": x_top,
            "y1": y_top,
            "x2": x_bot,
            "y2": y_bot,
            "length": float(span),
            "slope": float(slope),
            "coverage": float(coverage),
            "fill_coverage": float(fill_coverage),
            "fill_added_rows": int(fill_added_rows),
            "local_density": 0.0,
            "log_nfa": float(score),
            "mean_response": mean_response,
            "max_response": peak_response,
            "score": float(score),
            "repair_width": int(max(1, repair_width)),
            "votes": 1,
            "support_x": fill_xs_arr,
            "support_y": fill_ys_arr,
        })

    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:max_lines]


def _detect_spatial_lines(frame: np.ndarray, p: Dict, preset_mask: Optional[np.ndarray]) -> Tuple[List[Dict], Dict]:
    gray = _to_gray(frame)
    h, w = gray.shape
    valid = _build_valid_mask((h, w), p, preset_mask)

    out_lines: List[Dict] = []
    debug: Dict[str, int | float] = {}

    for kind in ("thin", "soft"):
        if kind == "soft" and not p.get("enable_soft_detection", True):
            debug["soft_lines"] = 0
            continue

        prof = _profile_response(gray, valid, p, kind)
        seed = prof["seed"] & valid
        seed = _prune_seed(seed, prof["response"], p, kind)
        density = _local_density(seed, p, kind)
        raw_lines = _hough_candidates(seed, p, kind)
        evaluated: List[Dict] = []
        for raw in raw_lines:
            item = _evaluate_line(raw, prof, seed, density, valid, p, kind)
            if item is not None:
                evaluated.append(item)

        lane_lines: List[Dict] = []
        emulsion_lines: List[Dict] = []
        emulsion_track_lines: List[Dict] = []
        if kind == "soft":
            lane_lines = _soft_lane_recovery(prof, seed, valid, p)
            emulsion_lines = _irregular_emulsion_damage_recovery(prof, valid, p)
            emulsion_track_lines = _emulsion_vertical_track_recovery(prof, valid, p)
            evaluated.extend(lane_lines)
            evaluated.extend(emulsion_lines)
            evaluated.extend(emulsion_track_lines)

        evaluated = _suppress_duplicate_lines(evaluated, p, kind)
        out_lines.extend(evaluated)

        debug[f"{kind}_seed_px"] = int(np.sum(seed))
        debug[f"{kind}_raw_lines"] = len(raw_lines)
        debug[f"{kind}_lane_lines"] = len(lane_lines)
        debug[f"{kind}_emulsion_lines"] = len(emulsion_lines)
        debug[f"{kind}_emulsion_track_lines"] = len(emulsion_track_lines)
        debug[f"{kind}_lines"] = len(evaluated)
        debug[f"{kind}_abs_thr"] = float(prof["abs_thr"])
        debug[f"{kind}_rel_thr"] = float(prof["rel_thr"])

    return out_lines, debug


# -----------------------------------------------------------------------------
# Temporal validation and mask construction
# -----------------------------------------------------------------------------


def _temporal_validate(line_lists: Sequence[List[Dict]], p: Dict) -> List[Dict]:
    if not line_lists:
        return []
    center_idx = len(line_lists) // 2
    center_lines = line_lists[center_idx]
    accepted: List[Dict] = []
    min_votes = int(p.get("temporal_min_votes", 2))
    neighbor_radius = int(p.get("temporal_neighbor_radius", 2))

    for line in center_lines:
        votes = 1
        best_neighbor_score = 0.0
        for j, lines in enumerate(line_lists):
            if j == center_idx:
                continue
            dist = abs(j - center_idx)
            if dist > neighbor_radius:
                continue
            for other in lines:
                # Thin can match soft and vice versa, but similar kind is preferred by score.
                if _same_lane(line, other, p, frame_distance=dist):
                    votes += 1
                    best_neighbor_score = max(best_neighbor_score, float(other.get("score", 0.0)))
                    break

        line = dict(line)
        line["votes"] = votes
        # Very strong lines can pass with fewer votes, but ordinary soft lines need temporal support.
        strong = (
            line["score"] >= float(p.get("temporal_strong_score", 30.0)) and
            line["mean_response"] >= float(p.get("temporal_strong_mean_response", 0.0045))
        )
        if votes >= min_votes or strong:
            line["score"] += 2.0 * votes + 0.02 * best_neighbor_score
            accepted.append(line)

    accepted.sort(key=lambda d: d["score"], reverse=True)
    return accepted


def _draw_line_mask(mask: np.ndarray, line: Dict, width: int) -> None:
    x1, y1, x2, y2 = int(round(line["x1"])), int(round(line["y1"])), int(round(line["x2"])), int(round(line["y2"]))
    width = max(1, int(width))
    cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=width, lineType=cv2.LINE_AA)


def _draw_supported_line_mask(mask: np.ndarray, line: Dict, width: int, max_bridge_gap: int) -> None:
    """
    Draw only the supported parts of a scratch hypothesis.

    This is the key v3 change.  A Hough line is not trusted as the repair
    geometry.  We draw the detected support points and bridge only short
    vertical gaps.  This prevents full diagonal false-positive bands and helps
    intermittent scratches without forcing full-height repair.
    """
    xs = line.get("support_x", None)
    ys = line.get("support_y", None)

    if xs is None or ys is None or len(xs) == 0:
        _draw_line_mask(mask, line, width)
        return

    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    order = np.argsort(ys)
    xs = xs[order]
    ys = ys[order]

    h, w = mask.shape
    width = max(1, int(width))
    radius = max(1, width // 2)
    max_bridge_gap = max(0, int(max_bridge_gap))

    # Draw support points.
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (int(x), int(y)), radius, 1, -1)

    # Bridge only short gaps.  Longer gaps remain unrepaired unless there is
    # actual support in another accepted segment.
    for i in range(len(xs) - 1):
        gap = int(ys[i + 1] - ys[i])
        if gap <= 0:
            continue
        if gap <= max_bridge_gap:
            cv2.line(
                mask,
                (int(xs[i]), int(ys[i])),
                (int(xs[i + 1]), int(ys[i + 1])),
                1,
                thickness=width,
                lineType=cv2.LINE_AA,
            )


def _budget_lines(lines: List[Dict], shape: Tuple[int, int], p: Dict, kind: str) -> Tuple[np.ndarray, List[Dict]]:
    h, w = shape
    max_fraction = float(p.get("thin_max_mask_fraction" if kind == "thin" else "soft_max_mask_fraction", 0.004 if kind == "thin" else 0.003))
    limit = int(max_fraction * h * w)
    mask = np.zeros((h, w), dtype=np.uint8)
    kept: List[Dict] = []
    for line in sorted(lines, key=lambda d: d["score"], reverse=True):
        tmp = mask.copy()
        bridge_gap = int(p.get("thin_bridge_gap", 18) if kind == "thin" else p.get("soft_bridge_gap", 42))
        _draw_supported_line_mask(
            tmp,
            line,
            int(line.get("repair_width", 3 if kind == "thin" else 13)),
            bridge_gap,
        )
        if int(np.sum(tmp > 0)) <= limit or len(kept) == 0:
            mask = tmp
            kept.append(line)
        else:
            # Stop once the protected budget is full; later lines have lower score.
            continue
    return mask.astype(bool), kept


def _postprocess_mask(mask: np.ndarray, p: Dict) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    # Light vertical close only. Do not create full-height bands.
    close_h = _odd(p.get("final_close_height", 5), 1)
    close_w = _odd(p.get("final_close_width", 1), 1)
    if close_h > 1 or close_w > 1:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
            iterations=1,
        ).astype(bool)
    return mask.astype(bool)



def _soft_track_bridges(lines: List[Dict], shape: Tuple[int, int], p: Dict) -> List[Dict]:
    """
    True soft-track fill.

    Earlier versions bridged some gaps, but still treated each accepted soft
    segment too independently.  v9 groups accepted soft_line/soft_lane segments
    into physical scratch tracks, fits a piecewise x(y) path, and fills rows
    between confirmed anchors.

    Important safety constraints:
    - It only starts from accepted soft lines/lanes that already passed spatial
      and temporal validation.
    - It does not use raw seed pixels.
    - It does not create full-frame columns; it fills only between anchors that
      belong to the same validated track.
    - It rejects tracks with excessive horizontal wandering.
    """
    if not bool(p.get("enable_soft_track_fill", p.get("enable_soft_track_bridging", True))):
        return []

    h, w = shape

    # v10: use accepted thin scratches as optional anchors for the physical
    # track, but only produce a soft track if the group contains at least one
    # accepted soft line/lane.  This helps fill gaps where the visible soft
    # scratch only generated thin centerline anchors.
    use_thin_anchors = bool(p.get("soft_track_use_thin_anchors", True))
    candidates = [
        l for l in lines
        if l.get("kind") == "soft" or (use_thin_anchors and l.get("kind") == "thin")
    ]
    if not candidates:
        return []

    items = []
    for idx, line in enumerate(candidates):
        xs = np.asarray(line.get("support_x", []), dtype=np.float32)
        ys = np.asarray(line.get("support_y", []), dtype=np.float32)
        if xs.size < 2 or ys.size < 2:
            continue

        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        if y_max <= y_min:
            continue

        items.append({
            "idx": idx,
            "line": line,
            "xs": xs,
            "ys": ys,
            "x_med": float(np.median(xs)),
            "y_min": y_min,
            "y_max": y_max,
            "slope": float(line.get("slope", 0.0)),
            "score": float(line.get("score", 0.0)),
            "source": str(line.get("source", "soft_line")),
        })

    if not items:
        return []

    n = len(items)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    x_tol = float(p.get("soft_track_group_x_tol", 34.0))
    slope_tol = float(p.get("soft_track_group_slope_tol", 0.16))
    group_y_gap = int(p.get("soft_track_group_y_gap", 900))

    # Graph clustering: connect segments if their predicted x is close at their
    # overlap/nearest vertical position and their slopes are compatible.
    for i in range(n):
        a = items[i]
        for j in range(i + 1, n):
            b = items[j]

            # Vertical gap between segment spans.
            y_gap = max(0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
            if y_gap > group_y_gap:
                continue

            if abs(a["slope"] - b["slope"]) > slope_tol:
                continue

            # Test x distance at overlap midpoint, or at midpoint of the gap.
            y0 = max(a["y_min"], b["y_min"])
            y1 = min(a["y_max"], b["y_max"])
            if y1 >= y0:
                y_test = 0.5 * (y0 + y1)
            else:
                y_test = 0.5 * (min(a["y_max"], b["y_max"]) + max(a["y_min"], b["y_min"]))

            ax = _line_x_at_y(a["line"], y_test)
            bx = _line_x_at_y(b["line"], y_test)

            # Allow a little extra tolerance over larger vertical gaps because
            # physical scratches can drift.
            tol = x_tol + float(p.get("soft_track_gap_x_slope_allowance", 0.015)) * float(y_gap)
            if abs(ax - bx) <= tol:
                union(i, j)

    groups_by_root: Dict[int, List[Dict]] = {}
    for i, item in enumerate(items):
        groups_by_root.setdefault(find(i), []).append(item)

    out: List[Dict] = []

    min_segments = int(p.get("soft_track_min_segments", 1))
    single_min_span = int(p.get("soft_track_single_min_span", 260))
    min_span = int(p.get("soft_track_min_span", 160))
    min_anchor_rows = int(p.get("soft_track_min_anchor_rows", 14))
    max_x_mad = float(p.get("soft_track_max_x_mad", 12.0))
    max_x_span = float(p.get("soft_track_max_x_span", 72.0))
    max_slope = float(p.get("soft_track_max_abs_slope", p.get("soft_max_abs_slope", 0.18)))
    bridge_gap = int(p.get("soft_track_bridge_gap", 900))
    max_tracks = int(p.get("soft_track_max_tracks", 16))
    bin_size = int(p.get("soft_track_bin_size", 18))

    track_min_w = int(p.get("soft_track_min_repair_width", p.get("soft_line_min_repair_width", 23)))
    track_max_w = max(track_min_w, int(p.get("soft_max_repair_width", 31)))
    repair_width = int(p.get("soft_track_repair_width", track_max_w))
    repair_width = int(np.clip(
        repair_width,
        track_min_w,
        track_max_w,
    ))
    if repair_width % 2 == 0:
        repair_width += 1

    for group in groups_by_root.values():
        all_x = []
        all_y = []
        all_scores = []
        source_count = len(group)
        soft_count = sum(1 for item in group if item["line"].get("kind") == "soft")

        # Do not widen purely thin scratch tracks.  Thin scratches are already
        # handled well by the thin path.
        if soft_count <= 0:
            continue

        for item in group:
            all_x.append(item["xs"])
            all_y.append(item["ys"])
            all_scores.append(item["score"])

        xs = np.concatenate(all_x).astype(np.float32)
        ys = np.concatenate(all_y).astype(np.float32)

        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        span = y_max - y_min + 1

        # Accept a single accepted soft line as a track only if it is already
        # long enough.  This helps when Hough returns one long but dotted line.
        if source_count < min_segments and span < single_min_span:
            continue
        if span < min_span:
            continue

        if xs.size < min_anchor_rows:
            continue

        # Robust track-shape rejection.
        x_med = float(np.median(xs))
        x_mad = float(np.median(np.abs(xs - x_med)))
        x_span = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        if x_mad > max_x_mad:
            continue
        if x_span > max_x_span:
            continue

        # Build robust anchor bins.  This makes the track path piecewise and
        # tolerant of horizontal drift, instead of forcing one global straight line.
        bins: Dict[int, List[Tuple[float, float]]] = {}
        for xv, yv in zip(xs, ys):
            key = int(round(float(yv) / max(1, bin_size)))
            bins.setdefault(key, []).append((float(xv), float(yv)))

        anchor_y = []
        anchor_x = []
        for key in sorted(bins.keys()):
            vals = bins[key]
            if not vals:
                continue
            bx = np.array([v[0] for v in vals], dtype=np.float32)
            by = np.array([v[1] for v in vals], dtype=np.float32)
            anchor_x.append(float(np.median(bx)))
            anchor_y.append(float(np.median(by)))

        if len(anchor_y) < 2:
            continue

        anchor_y_arr = np.asarray(anchor_y, dtype=np.float32)
        anchor_x_arr = np.asarray(anchor_x, dtype=np.float32)

        # Approximate global slope for safety/debug.
        if np.ptp(anchor_y_arr) > 1:
            slope, intercept = np.polyfit(anchor_y_arr, anchor_x_arr, 1)
        else:
            slope = 0.0
            intercept = float(np.median(anchor_x_arr))

        if abs(float(slope)) > max_slope:
            continue

        rows = np.arange(y_min, y_max + 1, dtype=np.int32)
        interp_x = np.interp(rows.astype(np.float32), anchor_y_arr, anchor_x_arr)

        support_x = []
        support_y = []

        # Fill rows only between nearby anchors.  This is the true gap bridge.
        # If a gap between anchors is larger than bridge_gap, that section stays
        # open, preventing arbitrary full-height repair.
        for yy, xx in zip(rows, interp_x):
            pos = np.searchsorted(anchor_y_arr, float(yy))
            if pos == 0 or pos >= len(anchor_y_arr):
                continue
            gap = float(anchor_y_arr[pos] - anchor_y_arr[pos - 1])
            if gap > bridge_gap:
                continue

            xi = int(round(float(xx)))
            if 0 <= xi < w:
                support_x.append(xi)
                support_y.append(int(yy))

        if len(support_y) < min_anchor_rows:
            continue

        fill_added = int(max(0, len(support_y) - len(np.unique(ys.astype(np.int32)))))
        score = float(np.mean(all_scores)) + 0.10 * len(support_y) + 20.0 + 2.0 * source_count

        out.append({
            "kind": "soft",
            "source": "soft_track",
            "x1": float(interp_x[0]),
            "y1": float(rows[0]),
            "x2": float(interp_x[-1]),
            "y2": float(rows[-1]),
            "length": float(len(support_y)),
            "slope": float(slope),
            "coverage": 1.0,
            "fill_added_rows": fill_added,
            "local_density": 0.0,
            "log_nfa": float(score),
            "mean_response": 0.0,
            "max_response": 0.0,
            "score": float(score),
            "repair_width": int(repair_width),
            "votes": 1,
            "support_x": np.asarray(support_x, dtype=np.int32),
            "support_y": np.asarray(support_y, dtype=np.int32),
        })

    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:max_tracks]

def _remove_edge_artifact_components(mask: np.ndarray, valid: np.ndarray, p: Dict) -> np.ndarray:
    """
    Remove remaining scratch-mask blobs close to the frame/preset-mask edge.

    This is intentionally component-based so real slender scratches near the
    edge can survive, while broad gate/mask-edge clusters are removed.
    """
    if not np.any(mask):
        return mask.astype(bool)
    if not bool(p.get("enable_edge_artifact_cleanup", True)):
        return mask.astype(bool)

    h, w = mask.shape
    m = mask.astype(np.uint8)

    # Distance from the current valid-mask boundary.
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    boundary_dist = float(p.get("edge_artifact_boundary_dist", 140.0))
    frame_strip = int(p.get("edge_artifact_frame_strip", 120))

    max_width_near_edge = int(p.get("edge_artifact_max_width", 54))
    min_area = int(p.get("edge_artifact_min_area", 250))
    max_fill = float(p.get("edge_artifact_max_fill", 0.28))
    min_aspect_keep = float(p.get("edge_artifact_min_aspect_keep", 4.0))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    removed = 0

    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = labels == i

        if area <= 0:
            continue

        near_mask_boundary = float(np.percentile(dist[comp], 10)) < boundary_dist
        near_frame_edge = (
            x < frame_strip or y < frame_strip or
            (x + cw) > (w - frame_strip) or (y + ch) > (h - frame_strip)
        )

        aspect = max(ch / max(cw, 1), cw / max(ch, 1))
        fill = area / max(1, cw * ch)

        # Keep slender scratches unless they are unusually wide/fat near edge.
        slender = aspect >= min_aspect_keep and cw <= max_width_near_edge

        remove = False
        if (near_mask_boundary or near_frame_edge) and area >= min_area:
            if not slender:
                remove = True
            if cw > max_width_near_edge:
                remove = True
            if fill > max_fill and area > min_area:
                remove = True

        if remove:
            removed += 1
            continue

        out[comp] = 1

    if removed:
        print(f"[EDGE CLEANUP] removed {removed} edge artifact component(s).")

    return out.astype(bool)



def detect_scratch_segments_stack(frames: Sequence[np.ndarray], p: Dict, preset_mask: Optional[np.ndarray] = None):
    """
    Detect scratches in the center frame of a temporal stack.

    Returns:
        mask: bool array for center frame
        debug: dict
    """
    if not frames:
        raise ValueError("detect_scratch_segments_stack requires at least one frame")

    # Detect spatial lines in each frame of the local temporal window.
    all_lines: List[List[Dict]] = []
    all_debug: List[Dict] = []
    for frame in frames:
        lines, dbg = _detect_spatial_lines(frame.astype(np.float32), p, preset_mask)
        all_lines.append(lines)
        all_debug.append(dbg)

    center_idx = len(frames) // 2
    h, w = frames[center_idx].shape[:2]
    valid = _build_valid_mask((h, w), p, preset_mask)

    accepted = _temporal_validate(all_lines, p)
    thin_lines = [l for l in accepted if l.get("kind") == "thin"]
    soft_lines = [l for l in accepted if l.get("kind") == "soft"]

    # v9: add true track-level soft fill between accepted soft segments on the
    # same scratch. This bridges missing intermittent sections without using raw
    # false-positive seed pixels.
    soft_track_lines = _soft_track_bridges(accepted, (h, w), p)
    if soft_track_lines:
        soft_lines = soft_lines + soft_track_lines

    thin_mask, thin_kept = _budget_lines(thin_lines, (h, w), p, "thin")
    soft_mask, soft_kept = _budget_lines(soft_lines, (h, w), p, "soft")

    # Soft must not override thin budget, but may overlap it.
    mask = (thin_mask | soft_mask) & valid

    # Final global cap as a last-resort safety. Prefer line-level budget, but
    # never return huge masks.
    max_fraction = float(p.get("scratch_max_mask_fraction", 0.008))
    if np.mean(mask) > max_fraction:
        # Keep all thin lines first, then add soft lines by score until global cap.
        safe = thin_mask.copy()
        limit = int(max_fraction * h * w)
        for line in sorted(soft_kept, key=lambda d: d["score"], reverse=True):
            tmp = safe.copy().astype(np.uint8)
            _draw_supported_line_mask(
                tmp,
                line,
                int(line.get("repair_width", 13)),
                int(p.get("soft_bridge_gap", 42)),
            )
            if int(np.sum(tmp > 0)) <= limit:
                safe = tmp.astype(bool)
        mask = safe & valid
        print(f"[AC SAFETY] global mask cap applied: {np.mean(mask) * 100:.3f}%")

    mask = _remove_edge_artifact_components(mask, valid, p) & valid
    mask = _postprocess_mask(mask, p) & valid
    mask = _remove_edge_artifact_components(mask, valid, p) & valid

    cdbg = all_debug[center_idx] if all_debug else {}
    print(
        "[AC PIXELS] "
        f"thin_seed={cdbg.get('thin_seed_px', 0)} soft_seed={cdbg.get('soft_seed_px', 0)} "
        f"thin_raw={cdbg.get('thin_raw_lines', 0)} soft_raw={cdbg.get('soft_raw_lines', 0)} "
        f"soft_lane={cdbg.get('soft_lane_lines', 0)} "
        f"emulsion={cdbg.get('soft_emulsion_lines', 0)} "
        f"emulsion_track={cdbg.get('soft_emulsion_track_lines', 0)}"
    )
    soft_lane_kept = sum(1 for l in soft_kept if l.get("source") == "soft_lane")
    soft_lane_fill_rows = sum(int(l.get("fill_added_rows", 0)) for l in soft_kept if l.get("source") == "soft_lane")
    soft_lane_widths = [int(l.get("repair_width", 0)) for l in soft_kept if l.get("source") == "soft_lane"]
    soft_line_fill_rows = sum(int(l.get("fill_added_rows", 0)) for l in soft_kept if l.get("source") == "soft_line")
    soft_line_widths = [int(l.get("repair_width", 0)) for l in soft_kept if l.get("source") == "soft_line"]
    soft_track_kept = sum(1 for l in soft_kept if l.get("source") == "soft_track")
    soft_track_fill_rows = sum(int(l.get("fill_added_rows", 0)) for l in soft_kept if l.get("source") == "soft_track")
    soft_track_widths = [int(l.get("repair_width", 0)) for l in soft_kept if l.get("source") == "soft_track"]
    emulsion_kept = sum(1 for l in soft_kept if l.get("source") == "emulsion_damage")
    emulsion_widths = [int(l.get("repair_width", 0)) for l in soft_kept if l.get("source") == "emulsion_damage"]
    emulsion_track_kept = sum(1 for l in soft_kept if l.get("source") == "emulsion_track")
    emulsion_track_fill_rows = sum(int(l.get("fill_added_rows", 0)) for l in soft_kept if l.get("source") == "emulsion_track")
    emulsion_track_widths = [int(l.get("repair_width", 0)) for l in soft_kept if l.get("source") == "emulsion_track"]
    print(
        "[AC LINES] "
        f"center={len(all_lines[center_idx])} accepted={len(accepted)} "
        f"thin={len(thin_kept)}/{len(thin_lines)} "
        f"soft={len(soft_kept)}/{len(soft_lines)} "
        f"soft_lane_kept={soft_lane_kept} "
        f"soft_lane_fill_rows={soft_lane_fill_rows} "
        f"soft_lane_widths={soft_lane_widths[:6]} "
        f"soft_line_fill_rows={soft_line_fill_rows} "
        f"soft_line_widths={soft_line_widths[:6]} "
        f"soft_track_kept={soft_track_kept} "
        f"soft_track_fill_rows={soft_track_fill_rows} "
        f"soft_track_widths={soft_track_widths[:6]} "
        f"emulsion_kept={emulsion_kept} "
        f"emulsion_widths={emulsion_widths[:6]} "
        f"emulsion_track_kept={emulsion_track_kept} "
        f"emulsion_track_fill_rows={emulsion_track_fill_rows} "
        f"emulsion_track_widths={emulsion_track_widths[:6]}"
    )
    print(
        "[SCRATCH MASK] "
        f"mask_px={int(np.sum(mask))}, mask_fraction={np.mean(mask) * 100:.3f}%"
    )

    debug = {
        "accepted": len(accepted),
        "thin_lines": len(thin_kept),
        "soft_lines": len(soft_kept),
        "mask_px": int(np.sum(mask)),
        "mask_fraction": float(np.mean(mask)),
        "center_debug": cdbg,
    }
    return mask.astype(bool), debug


# Backward-compatible wrapper used by older restore.py versions.
def detect_scratches_stack(frames: Sequence[np.ndarray], thresh=1.0, p: Optional[Dict] = None):
    if p is None:
        p = {}
    mask, _ = detect_scratch_segments_stack(frames, p)
    return mask.astype(np.float32)
