import numpy as np
import cv2


def auto_detect_mask(img):
    """
    Automatic film-gate detection.

    This is kept as a fallback only.
    It may fail on very dark scans because it relies on brightness.
    """
    gray = np.mean(img, axis=2)
    g = gray / (np.max(gray) + 1e-6)

    mask = g > 0.02

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    )

    return mask.astype(np.float32)


def load_preset_mask(path, target_shape):
    """
    Load a black/white preset mask from PNG/TIFF/JPG/BMP.

    White pixels mean: valid image area.
    Black pixels mean: excluded area.

    The mask is resized to match the EXR frame if necessary.
    """
    mask_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        raise ValueError(f"Could not read mask image: {path}")

    # If RGB/RGBA, convert to grayscale by averaging channels.
    if mask_img.ndim == 3:
        if mask_img.shape[2] == 4:
            mask_img = mask_img[:, :, :3]
        mask_img = np.mean(mask_img, axis=2)

    # Resize to match frame height/width.
    target_h, target_w = target_shape[:2]
    if mask_img.shape[:2] != (target_h, target_w):
        mask_img = cv2.resize(
            mask_img,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )

    mask_img = mask_img.astype(np.float32)

    max_value = np.max(mask_img)
    if max_value > 1.0:
        mask_img /= max_value

    mask = (mask_img >= 0.5).astype(np.float32)

    return mask
