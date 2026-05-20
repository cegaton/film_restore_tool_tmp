import numpy as np
import cv2


def keep_components_by_area(mask, min_area=12, max_area=None):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    out = np.zeros(mask.shape, dtype=np.uint8)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        out[labels == i] = 1

    return out.astype(bool)


def detect_dust(frames, center, thresh):
    """
    Detect dust using aligned neighboring frames.

    Small dust:
        center differs from several neighbors.

    Large dust:
        center differs from temporal median.
    """

    if len(frames) == 0:
        return np.zeros(center.shape[:2], dtype=np.float32)

    stack = np.stack(frames, axis=0)
    temporal_median = np.median(stack, axis=0)

    # --- Small dust / specks ---
    diffs = np.stack(
        [np.mean(np.abs(center - f), axis=2) for f in frames],
        axis=0
    )

    votes = np.sum(diffs > thresh, axis=0)
    small = votes >= max(1, min(2, len(frames)))

    small = keep_components_by_area(small, min_area=3)

    # --- Large irregular dust ---
    med_diff = np.mean(np.abs(center - temporal_median), axis=2)

    large = med_diff > max(thresh * 1.8, 0.040)

    large = cv2.morphologyEx(
        large.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )

    large = cv2.morphologyEx(
        large,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    large = keep_components_by_area(large, min_area=40,max_area=12000)

    combined = small | large

    # Expand to include feathered dust borders.
    combined = cv2.dilate(
        combined.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    return combined.astype(np.float32)
