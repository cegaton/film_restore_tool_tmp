import numpy as np
import cv2

# -----------------------------------------------------------------------------
# scratch_clean_v7.py
#
# Detection only. Designed for scanned negative film where line scratches appear
# as BRIGHT/WHITE near-vertical defects.
#
# Main change from previous attempts:
#   - no full-height profile bands;
#   - no global column repair masks;
#   - detection is local in x AND y;
#   - candidate pixels must be narrow bright ridges with local vertical support;
#   - candidate segments are validated against their LOCAL background density,
#     so textured/bright image areas do not dominate the scratch budget.
# -----------------------------------------------------------------------------


def _to_gray(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return np.mean(img.astype(np.float32), axis=2)


def _odd(v, minimum=3):
    v = int(round(v))
    v = max(minimum, v)
    if v % 2 == 0:
        v += 1
    return v


def _sample_shifted(arr, offset):
    h, w = arr.shape
    off = int(offset)
    pad = abs(off) + 8
    padded = cv2.copyMakeBorder(arr, 0, 0, pad, pad, cv2.BORDER_REFLECT_101)
    start = pad + off
    return padded[:, start:start + w]


def _build_valid_mask(shape, p, preset_mask=None):
    h, w = shape
    valid = np.ones((h, w), dtype=bool)

    if preset_mask is not None:
        pm = preset_mask
        if pm.ndim == 3:
            pm = pm[:, :, 0]
        valid &= pm > 0.5

        erode_px = int(p.get("mask_edge_erode", 12))
        if erode_px > 0:
            k = erode_px * 2 + 1
            valid = cv2.erode(
                valid.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                iterations=1,
            ).astype(bool)

    border = int(p.get("ignore_frame_border_px", 24))
    if border > 0:
        valid[:, :border] = False
        valid[:, max(0, w - border):] = False
        valid[:border, :] = False
        valid[max(0, h - border):, :] = False

    return valid


def _estimate_slope_from_component(comp):
    ys, xs = np.where(comp)
    if xs.size < 12:
        return 0.0
    try:
        return float(np.polyfit(ys.astype(np.float32), xs.astype(np.float32), 1)[0])
    except Exception:
        return 0.0


def _horizontal_profile_response(gray, valid, p):
    """
    Bright-ridge response for negative-film scratches.

    The central stripe must be brighter than its immediate left/right
    neighborhoods. Side-coherence is measured, but not used as a hard veto for
    every pixel because real scratches can sit over uneven scene content.
    """
    gray_s = gray.astype(np.float32)
    if int(p.get("scratch_prefilter", 1)):
        gray_s = cv2.GaussianBlur(gray_s, (3, 3), 0)

    side_width = _odd(p.get("profile_side_width", 5), 3)
    gap = int(p.get("profile_side_gap", 2))
    widths = p.get("profile_widths", (1, 3, 5))

    best_response = np.zeros(gray.shape, dtype=np.float32)
    best_bg = np.maximum(gray_s, 0.0).astype(np.float32)
    best_left = np.zeros(gray.shape, dtype=np.float32)
    best_right = np.zeros(gray.shape, dtype=np.float32)
    best_side_diff = np.full(gray.shape, np.inf, dtype=np.float32)
    best_width = np.ones(gray.shape, dtype=np.float32)

    side_mean = cv2.blur(gray_s, (side_width, 1))

    for width in widths:
        width = _odd(width, 1)
        center = gray_s if width <= 1 else cv2.blur(gray_s, (width, 1))

        shift = gap + width // 2 + side_width // 2
        left = _sample_shifted(side_mean, -shift)
        right = _sample_shifted(side_mean, shift)

        # Use max(left,right), not average. This rejects ordinary bright edges.
        bg = np.maximum(left, right)
        response = np.maximum(center - bg, 0.0)
        side_diff = np.abs(left - right)

        better = response > best_response
        best_response[better] = response[better]
        best_bg[better] = bg[better]
        best_left[better] = left[better]
        best_right[better] = right[better]
        best_side_diff[better] = side_diff[better]
        best_width[better] = width

    rel = best_response / np.maximum(best_bg, float(p.get("scratch_rel_floor", 0.025)))

    # Robust-ish local texture estimate. Textured/bright faces get a higher bar.
    noise_bg_w = _odd(p.get("profile_noise_bg_width", 31), 5)
    noise_box = _odd(p.get("profile_local_noise_box", 51), 9)
    hblur = cv2.blur(gray_s, (noise_bg_w, 1))
    hp = np.abs(gray_s - hblur)
    local_noise = cv2.blur(hp, (noise_box, noise_box))

    abs_thr = float(p.get("scratch_abs", 0.0075))
    rel_thr = float(p.get("scratch_rel", 0.12))
    noise_mul = float(p.get("scratch_noise_mul", 1.75))

    adaptive_abs_thr = np.maximum(abs_thr, local_noise * noise_mul)

    # Soft side coherence. Later component scoring penalizes large side_diff.
    side_base = float(p.get("profile_side_coherence", 0.018))
    side_rel = float(p.get("profile_side_coherence_rel", 0.30))
    side_noise_mul = float(p.get("profile_side_noise_mul", 1.75))
    side_thr = side_base + side_rel * best_response + side_noise_mul * local_noise
    coherent = best_side_diff <= side_thr

    # Narrow local horizontal peak test. This is crucial: broad image highlights
    # should not become scratch seeds.
    peak_w = _odd(p.get("scratch_peak_width", 9), 3)
    resp_max = cv2.dilate(
        best_response,
        cv2.getStructuringElement(cv2.MORPH_RECT, (peak_w, 1)),
        iterations=1,
    )
    local_peak = best_response >= (resp_max - float(p.get("scratch_peak_eps", 1e-5)))

    strict_seed = (
        (best_response >= adaptive_abs_thr) &
        (rel >= rel_thr) &
        local_peak &
        valid
    )

    # Strong seeds may violate side coherence but must be very bright and peak-like.
    strong_seed = (
        (best_response >= float(p.get("scratch_strong_abs", 0.030))) &
        (rel >= float(p.get("scratch_strong_rel", 0.12))) &
        local_peak &
        valid
    )

    # Coherence is required for weak seeds, optional for strong ones.
    seed = (strict_seed & coherent) | strong_seed

    return {
        "gray_s": gray_s,
        "response": best_response,
        "rel": rel,
        "side_diff": best_side_diff,
        "width_map": best_width,
        "local_noise": local_noise,
        "coherent": coherent,
        "seed": seed,
        "abs_thr": abs_thr,
        "rel_thr": rel_thr,
    }


def _local_vertical_support_candidates(prof, valid, p):
    """
    Build a local candidate mask. This is NOT a full-column detector.

    A pixel is considered only if:
    - it is a bright narrow seed;
    - it lies on a local vertical support ridge;
    - that vertical support is stronger than neighbouring columns at the same y.
    """
    seed = prof["seed"] & valid
    response = prof["response"]

    close_seed_w = int(p.get("support_seed_x_dilate", 1))
    if close_seed_w > 1:
        seed_support = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(close_seed_w, 1), 1)),
            iterations=1,
        ).astype(bool)
    else:
        seed_support = seed

    vert_h = _odd(p.get("support_vertical_blur", 81), 9)
    support = cv2.blur(seed_support.astype(np.float32), (1, vert_h))

    bg_w = _odd(p.get("support_bg_width", 81), 15)
    support_bg = cv2.blur(support, (bg_w, 1))
    support_excess = support - support_bg

    # Local threshold, not global column threshold.
    min_support = float(p.get("support_min", 0.020))
    min_excess = float(p.get("support_excess", 0.010))
    rel_factor = float(p.get("support_rel_factor", 1.65))

    support_ok = (
        (support >= min_support) &
        (support_excess >= min_excess) &
        (support >= support_bg * rel_factor)
    )

    # Keep only original seed pixels; support only validates them.
    candidate = seed & support_ok & valid

    # A few very bright isolated scratches can have sparse seeds; allow them if
    # response is strong and vertical support is moderate.
    strong_candidate = (
        seed &
        (response >= float(p.get("scratch_strong_abs", 0.030))) &
        (support >= float(p.get("support_strong_min", 0.010))) &
        valid
    )
    candidate |= strong_candidate

    if p.get("debug_repair", False):
        print(
            f"[SCRATCH SUPPORT] seed_px={int(np.sum(seed))}, "
            f"candidate_px={int(np.sum(candidate))}, "
            f"support_max={float(np.max(support)):.3f}, "
            f"excess_max={float(np.max(support_excess)):.3f}"
        )

    return candidate, support, support_excess


def _local_background_seed_density(seed, comp_mask, bbox, p):
    h, w = seed.shape
    x0, y0, x1, y1 = bbox
    pad_x = int(p.get("segment_bg_pad_x", 50))
    pad_y = int(p.get("segment_bg_pad_y", 30))
    guard_x = int(p.get("segment_bg_guard_x", 8))

    xa = max(0, x0 - pad_x)
    xb = min(w, x1 + pad_x)
    ya = max(0, y0 - pad_y)
    yb = min(h, y1 + pad_y)

    region = seed[ya:yb, xa:xb]
    if region.size == 0:
        return 0.0

    # Exclude a guard band around the candidate scratch.
    xx0 = max(0, x0 - guard_x - xa)
    xx1 = min(xb - xa, x1 + guard_x - xa)
    bg_region = region.copy()
    bg_region[:, xx0:xx1] = False

    if bg_region.size == 0:
        return 0.0
    return float(np.mean(bg_region))


def _segments_from_candidate_mask(candidate, prof, valid, p):
    """Create validated scratch segments from the local candidate mask."""
    response = prof["response"]
    rel = prof["rel"]
    side_diff = prof["side_diff"]
    width_map = prof["width_map"]
    seed = prof["seed"] & valid

    close_h = _odd(p.get("segment_connect_height", 31), 3)
    close_w = _odd(p.get("segment_connect_width", 3), 1)
    work = candidate.astype(np.uint8)
    if close_h > 1 or close_w > 1:
        work = cv2.morphologyEx(
            work,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        )

    work = work.astype(bool) & valid

    num, labels, stats, _ = cv2.connectedComponentsWithStats(work.astype(np.uint8), connectivity=8)

    h, w = work.shape
    min_area = int(p.get("segment_min_area", 6))
    min_height = int(p.get("segment_min_height", 28))
    max_width = int(p.get("segment_max_width", 22))
    min_aspect = float(p.get("segment_min_aspect", 2.8))
    max_slope = float(p.get("segment_max_abs_slope", 0.75))
    min_z = float(p.get("segment_min_local_z", 3.0))
    min_density_ratio = float(p.get("segment_min_density_ratio", 2.0))
    max_area = int(h * w * float(p.get("segment_max_component_fraction", 0.0008)))

    segments = []
    rejected = 0

    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area or ch < min_height or cw > max_width:
            rejected += 1
            continue

        aspect = ch / max(1, cw)
        if aspect < min_aspect:
            rejected += 1
            continue

        comp = labels[y:y + ch, x:x + cw] == i
        slope = _estimate_slope_from_component(comp)
        if abs(slope) > max_slope:
            rejected += 1
            continue

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y:y + ch, x:x + cw][comp] = True

        # Local density validation, inspired by the a-contrario idea: accept only
        # when this segment is unlikely relative to local seed density.
        l = max(1, ch * max(1, min(cw, int(p.get("segment_line_width_for_density", 5)))))
        k = int(np.sum(seed[y:y + ch, x:x + cw][comp]))
        seg_density = k / max(1, area)
        p_bg = _local_background_seed_density(seed, full_mask, (x, y, x + cw, y + ch), p)
        p_bg = float(np.clip(p_bg, 1e-5, 0.95))
        # Conservative z-score using component area as sample size.
        z = (seg_density - p_bg) / np.sqrt(max(1e-6, p_bg * (1.0 - p_bg) / max(1, area)))
        density_ratio = seg_density / max(p_bg, 1e-5)

        resp_vals = response[y:y + ch, x:x + cw][comp]
        rel_vals = rel[y:y + ch, x:x + cw][comp]
        side_vals = side_diff[y:y + ch, x:x + cw][comp]
        width_vals = width_map[y:y + ch, x:x + cw][comp]

        mean_response = float(np.mean(resp_vals)) if resp_vals.size else 0.0
        max_response = float(np.max(resp_vals)) if resp_vals.size else 0.0
        mean_rel = float(np.mean(rel_vals)) if rel_vals.size else 0.0
        mean_side_diff = float(np.mean(side_vals)) if side_vals.size else 0.0

        # Strong clear scratches may pass with lower z; faint scratches need
        # local improbability and length.
        strong = max_response >= float(p.get("segment_strong_response", 0.030))
        if not strong:
            if z < min_z or density_ratio < min_density_ratio:
                rejected += 1
                continue

        score = (
            z * 0.010 +
            mean_response * np.sqrt(ch) * min(2.5, aspect / 3.0) +
            0.35 * max_response +
            0.010 * np.log1p(ch) -
            0.20 * mean_side_diff
        )

        segments.append({
            "bbox": (x, y, x + cw, y + ch),
            "center_x": x + cw * 0.5,
            "center_y": y + ch * 0.5,
            "width": cw,
            "height": ch,
            "area": area,
            "aspect": aspect,
            "slope": slope,
            "mean_response": mean_response,
            "max_response": max_response,
            "mean_rel": mean_rel,
            "mean_side_diff": mean_side_diff,
            "local_bg_density": p_bg,
            "segment_density": seg_density,
            "local_z": float(z),
            "density_ratio": float(density_ratio),
            "estimated_width": float(np.median(width_vals)) if width_vals.size else 1.0,
            "score": float(score),
            "mask": full_mask,
            "label": "local vertical scratch segment",
        })

    return segments, rejected, int(np.sum(work))


def detect_frame_scratch_segments(frame, p, preset_mask=None):
    gray = _to_gray(frame)
    h, w = gray.shape
    valid = _build_valid_mask((h, w), p, preset_mask=preset_mask)

    prof = _horizontal_profile_response(gray, valid, p)
    candidate, support, support_excess = _local_vertical_support_candidates(prof, valid, p)
    segments, rejected, raw_px = _segments_from_candidate_mask(candidate, prof, valid, p)

    if p.get("debug_repair", False):
        print(
            f"[SCRATCH DETECT] kept={len(segments)}, rejected={rejected}, "
            f"raw_px={raw_px}, abs_thr={prof['abs_thr']:.5f}, rel_thr={prof['rel_thr']:.5f}"
        )

    return segments


def _y_overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    overlap = max(0, min(ay1, by1) - max(ay0, by0))
    denom = max(1, min(ay1 - ay0, by1 - by0))
    return overlap / denom


def _best_match_count(center_seg, other_segments, frame_distance, p):
    max_dx = float(p.get("track_max_dx_per_frame", 18.0)) * max(1, frame_distance)
    min_overlap = float(p.get("track_min_y_overlap", 0.15))
    max_width_ratio = float(p.get("track_max_width_ratio", 4.0))
    max_slope_delta = float(p.get("track_max_slope_delta", 0.60))

    for seg in other_segments:
        if abs(center_seg["center_x"] - seg["center_x"]) > max_dx:
            continue
        if _y_overlap_ratio(center_seg, seg) < min_overlap:
            continue
        wr = max(center_seg["width"], seg["width"]) / max(1, min(center_seg["width"], seg["width"]))
        if wr > max_width_ratio:
            continue
        if abs(center_seg.get("slope", 0.0) - seg.get("slope", 0.0)) > max_slope_delta:
            continue
        return 1
    return 0


def select_center_scratch_segments(segment_lists, p):
    n = len(segment_lists)
    c_idx = n // 2
    center_segments = segment_lists[c_idx]
    selected = []

    min_votes = int(p.get("track_min_votes", 2))
    strong_score = float(p.get("track_strong_score", 0.060))
    strong_max = float(p.get("track_strong_max_response", 0.035))
    strong_z = float(p.get("track_strong_local_z", 6.0))
    max_neighbors = int(p.get("track_neighbor_radius", n // 2))

    for seg in center_segments:
        votes = 0
        for j, others in enumerate(segment_lists):
            if j == c_idx:
                continue
            dist = abs(j - c_idx)
            if dist > max_neighbors:
                continue
            votes += _best_match_count(seg, others, dist, p)

        strong = (
            seg["score"] >= strong_score or
            seg["max_response"] >= strong_max or
            seg.get("local_z", 0.0) >= strong_z
        )

        if votes >= min_votes or strong:
            ss = dict(seg)
            ss["track_votes"] = votes
            ss["strong"] = bool(strong)
            selected.append(ss)

    if p.get("debug_repair", False):
        print(
            f"[SCRATCH TRACK] center_candidates={len(center_segments)}, "
            f"selected={len(selected)}, min_votes={min_votes}"
        )

    return selected


def build_repair_mask_from_segments(segments, shape, p):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    base_width = int(p.get("scratch_repair_width", 1))
    extra = int(p.get("scratch_repair_extra_width", 0))
    max_width = min(
        int(p.get("scratch_repair_max_width", 3)),
        int(p.get("scratch_repair_hard_max_width", 3)),
    )

    for seg in segments:
        sm = seg["mask"].astype(bool)
        est = int(round(seg.get("estimated_width", seg.get("width", 1))))
        width = max(base_width, est + extra)
        width = min(width, max_width)
        if width % 2 == 0:
            width += 1

        if width > 1:
            sm = cv2.dilate(
                sm.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)),
                iterations=1,
            ).astype(bool)

        close_h = int(p.get("scratch_repair_close_height", 3))
        if close_h > 1:
            if close_h % 2 == 0:
                close_h += 1
            sm = cv2.morphologyEx(
                sm.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h)),
            ).astype(bool)

        mask |= sm

    return mask


def _budget_mask_by_segments(segments, shape, p, target_fraction):
    h, w = shape
    target_px = int(h * w * float(target_fraction))
    if target_px <= 0:
        return np.zeros((h, w), dtype=bool)

    ranked = sorted(
        segments,
        key=lambda s: (
            s.get("track_votes", 0),
            s.get("local_z", 0.0),
            s.get("density_ratio", 0.0),
            s.get("height", 0),
            s.get("score", 0.0),
            s.get("max_response", 0.0),
            -s.get("mean_side_diff", 0.0),
        ),
        reverse=True,
    )

    out = np.zeros((h, w), dtype=bool)
    for seg in ranked:
        candidate = build_repair_mask_from_segments([seg], (h, w), p)
        new_out = out | candidate
        if np.sum(new_out) > target_px and np.any(out):
            continue
        out = new_out
        if np.sum(out) >= target_px:
            break

    return out


def detect_scratch_segments_stack(frames, p, preset_mask=None):
    if not frames:
        raise ValueError("detect_scratch_segments_stack requires at least one frame")

    segment_lists = [
        detect_frame_scratch_segments(f, p, preset_mask=preset_mask)
        for f in frames
    ]

    center_segments = select_center_scratch_segments(segment_lists, p)
    h, w = frames[len(frames) // 2].shape[:2]
    mask = build_repair_mask_from_segments(center_segments, (h, w), p)

    valid = _build_valid_mask((h, w), p, preset_mask=preset_mask)
    mask &= valid

    max_frac = float(p.get("scratch_max_mask_fraction", 0.0020))
    frac = float(np.mean(mask))
    if frac > max_frac:
        mask = _budget_mask_by_segments(center_segments, (h, w), p, max_frac)
        mask &= valid
        if p.get("debug_repair", False):
            print(
                f"[SCRATCH BUDGET] {frac * 100:.3f}% -> "
                f"{float(np.mean(mask)) * 100:.3f}% target={max_frac * 100:.3f}%"
            )

    debug = {
        "frame_segment_counts": [len(x) for x in segment_lists],
        "selected_segments": len(center_segments),
        "mask_pixels": int(np.sum(mask)),
        "mask_fraction": float(np.mean(mask)),
    }

    if p.get("debug_repair", False):
        print(
            f"[SCRATCH MASK] selected_segments={len(center_segments)}, "
            f"mask_px={debug['mask_pixels']}, "
            f"mask_fraction={debug['mask_fraction'] * 100:.3f}%"
        )

    return mask.astype(bool), debug


# Backward compatibility with the original project name.
def detect_scratches_stack(frames, thresh, p):
    pp = dict(p)
    if thresh is not None:
        # Older GUI called this threshold differently. Lower values mean more sensitive.
        s = float(thresh)
        pp["scratch_abs"] = float(p.get("scratch_abs", 0.0075)) * s
        pp["scratch_rel"] = float(p.get("scratch_rel", 0.12)) * s
    mask, _ = detect_scratch_segments_stack(frames, pp, preset_mask=pp.get("preset_mask", None))
    return mask.astype(np.float32)
