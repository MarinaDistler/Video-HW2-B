import cv2
import numpy as np

def overlay_mask_on_frame(frame, mask, alpha=0.5):
    """
    Наложение маски на кадр.
    """
    mask_color = np.zeros_like(frame)
    mask_color[..., 0] = (mask * 255).astype(np.uint8)  # красная маска
    overlay = cv2.addWeighted(frame.astype(np.uint8), 1.0, mask_color, alpha, 0)
    return overlay

