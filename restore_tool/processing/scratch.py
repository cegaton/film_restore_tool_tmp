import numpy as np
import cv2


def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return 0.6745 * (x - med) / mad


def detect_thin_bright_scratches(gray_stack, thresh):
    responses = []

    # Include taller kernels for long vertical scratches.
    scales = [9, 21, 41, 81, 121]

    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, s))
        resp = np.stack(
            [cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k) for g in gray_stack],
            axis=0
        )
        responses.append(resp)

    resp = np.maximum.reduce(responses)
    return resp > thresh

def reject_structural_vertical_edges(gray, mask, edge_ratio_limit=0.35):
    """
    Reject wide-scratch detections that look like real vertical image structure.

    The goal is to reduce false positives on things like:
    - columns
    - trees
    - door frames
    - windows
    - standing people
    - vertical shadows
    - architectural edges

    A real scratch is usually a narrow abnormal vertical mark.
    A real scene structure often has strong vertical edges around it.

    Parameters
    ----------
    gray : np.ndarray
        Single grayscale frame, float32, usually in range 0..1.

    mask : np.ndarray
        Candidate scratch mask. Can be bool, uint8, or float.

    edge_ratio_limit : float
        Lower = stricter rejection.
        Higher = more permissive.

        Suggested values:
            0.25 = very strict
            0.35 = balanced
            0.45 = permissive

    Returns
    -------
    np.ndarray
        Boolean mask with likely structural false positives removed.
    """

    # Always work with clean types.
    g = gray.astype(np.float32)
    m = mask.astype(bool)

    if not np.any(m):
        return m

    # Horizontal gradient detects vertical edges.
    # Strong vertical scene structures usually create strong horizontal gradients.
    grad_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)

    # Build a local neighborhood around all candidate scratch pixels.
    neighborhood = cv2.dilate(
        m.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        iterations=1
    ).astype(bool)

    surrounding = neighborhood & ~m

    if not np.any(surrounding):
        return m

    # Robust local edge threshold.
    surrounding_edges = grad_x[surrounding]

    med = np.median(surrounding_edges)
    mad = np.median(np.abs(surrounding_edges - med)) + 1e-6

    local_edge_level = med + 3.0 * mad

    strong_edges = grad_x > local_edge_level

    # Process each connected scratch candidate separately.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        m.astype(np.uint8),
        connectivity=8
    )

    out = np.zeros_like(m, dtype=bool)

    for i in range(1, num):
        comp = labels == i

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        # Very tiny pieces are not worth rejecting here.
        # Let other cleanup steps handle them.
        if area < 8:
            out[comp] = True
            continue

        # Look slightly around this component.
        x0 = max(0, x - 8)
        x1 = min(m.shape[1], x + w + 8)
        y0 = max(0, y - 2)
        y1 = min(m.shape[0], y + h + 2)

        comp_region = comp[y0:y1, x0:x1]
        edge_region = strong_edges[y0:y1, x0:x1]

        # Edges around the component, excluding the component itself.
        surrounding_edge_region = edge_region & ~comp_region

        total_support = np.sum(comp_region) + np.sum(surrounding_edge_region)

        if total_support <= 0:
            out[comp] = True
            continue

        edge_ratio = np.sum(surrounding_edge_region) / total_support

        # If there are too many strong surrounding vertical edges,
        # this is probably a real image structure, not a scratch.
        if edge_ratio < edge_ratio_limit:
            out[comp] = True

    return out


def detect_wide_vertical_scratches_single(gray, thresh, p=None):
    """
    Detect wide bright or dark vertical scratches.

    This looks for vertical bands that differ from the local horizontal
    neighborhood. It catches soft wide scratches better than top-hat alone.
    """
    
    if p is None:
        p = {}

    g = gray.astype(np.float32)

    # Normalize robustly.
    g = robust_zscore(g)

    masks = []

    for width in [7, 17, 35]:
        # Smooth vertically so long scratches become more coherent.
        vertical = cv2.blur(g, (1, 61))

        # Estimate local horizontal background.
        bg_width = max(31, width * 5)
        background = cv2.blur(vertical, (bg_width, 1))

        # Bright OR dark vertical deviation.
        response = np.abs(vertical - background)

        z = robust_zscore(response)

        # For this detector, threshold should be less aggressive than top-hat.
        m = z > max(3.0, thresh)

        # Connect broken vertical segments.
        m = cv2.morphologyEx(
            m.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (width, 41))
        )

        # Require some vertical continuity.
        continuity = cv2.blur(m.astype(np.float32), (1, 101))
        m = continuity > 0.18

        masks.append(m)

    mask = np.any(np.stack(masks, axis=0), axis=0)

    # Remove tiny accidental detections.
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    ).astype(bool)
    
    # Validate with horizontal scratch profile.
    if np.any(mask):
        mask = horizontal_profile_scratch_test(
            gray,
            mask,
            smed=0.035,
            savg=0.045
        )
    
    if np.any(mask):
        mask = reject_structural_vertical_edges(
            gray,
            mask,
            edge_ratio_limit=0.35
        )


    # OpenCV does not accept bool arrays for dilation.
    mask = mask.astype(np.uint8)
    
    # Expand slightly to cover soft scratch edges.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1
    )
    
    return mask.astype(bool)


def detect_scratches_stack(frames, thresh, p):
    stack = np.stack(frames, axis=0)
    gray = np.mean(stack, axis=3)

    std = np.std(gray, axis=(1, 2), keepdims=True) + 1e-6
    gray_norm = gray / std

    thin = detect_thin_bright_scratches(gray_norm, thresh)

    wide = np.zeros_like(thin, dtype=bool)

    center_idx = len(frames) // 2
    wide[center_idx] = detect_wide_vertical_scratches_single(
        gray[center_idx],
        thresh,
        p
    )

    masks = thin | wide

    # Important change:
    # For center-frame restoration, do not require the scratch to appear
    # in multiple frames. Scratches/dust are often transient.
    center_idx = len(frames) // 2
    center_mask = masks[center_idx]

    # Add weak temporal support, but do not depend on it.
    votes = np.sum(masks, axis=0)
    temporal_mask = votes >= max(2, len(frames) // 2)

    mask = center_mask & (temporal_mask | thin[center_idx])

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
    )

    return mask.astype(np.float32)
    
    
def horizontal_profile_scratch_test(gray, candidate_mask, smed=0.025, savg=0.08):
    """
    Validate scratch candidates using a horizontal profile test.

    A true scratch should be an abnormal narrow vertical feature compared
    to its horizontal neighborhood, while the left and right neighborhoods
    should be reasonably similar.

    smed:
        Minimum difference from horizontal median.

    savg:
        Maximum allowed difference between left and right neighborhoods.
        Lower = stricter, fewer false positives.
    """
    g = gray.astype(np.float32)

    # Mild denoise, similar in spirit to the paper's Gaussian prefilter.
    g = cv2.GaussianBlur(g, (3, 3), 0)

    h, w = g.shape

    # Horizontal median background.
    # Kernel is horizontal only.
    median_bg = cv2.medianBlur((np.clip(g, 0, 1) * 255).astype(np.uint8), 9)
    median_bg = median_bg.astype(np.float32) / 255.0

    median_diff = np.abs(g - median_bg)

    # Left and right horizontal averages.
    left = np.zeros_like(g)
    right = np.zeros_like(g)

    # Use pixels a few columns away from the candidate center.
    # This avoids including the scratch itself.
    for dx in [4, 5, 6]:
        left[:, dx:] += g[:, :-dx]
        right[:, :-dx] += g[:, dx:]

    left /= 3.0
    right /= 3.0

    side_diff = np.abs(left - right)

    valid = (
        candidate_mask.astype(bool)
        & (median_diff >= smed)
        & (side_diff <= savg)
    )

    return valid
    

