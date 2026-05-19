import numpy as np
import cv2


def load_preset_mask(path, target_shape):
    """
    Load a black/white preset mask from an image file.

    White = valid image area
    Black = excluded area

    The mask is resized to match the EXR frame if needed.
    """
    mask_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        raise ValueError(f"Could not read mask image: {path}")

    # Convert RGB/RGBA mask to grayscale.
    if mask_img.ndim == 3:
        if mask_img.shape[2] == 4:
            mask_img = mask_img[:, :, :3]
        mask_img = np.mean(mask_img, axis=2)

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
