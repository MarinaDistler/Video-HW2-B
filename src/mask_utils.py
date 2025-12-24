import numpy as np
import cv2
import os

def warp_mask(mask_prev, flow):
    """
    Перенос маски mask_prev на следующий кадр с помощью оптического потока.
    """
    h, w = mask_prev.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[...,0]).astype(np.float32)
    map_y = (grid_y - flow[...,1]).astype(np.float32)
    mask_warped = cv2.remap(mask_prev, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return mask_warped

def smooth_mask(mask_prev_smooth, mask_current_warped, beta=0.1):
    """
    Временное сглаживание масок.
    """
    mask_smooth = beta * mask_prev_smooth + (1 - beta) * mask_current_warped
    mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
    return mask_smooth

def compute_temporal_stability(masks):
    """
    Метрика временной стабильности: среднее изменение маски между кадрами.
    """
    masks = np.array(masks)  # (T, H, W)
    diffs = np.abs(masks[1:] - masks[:-1])
    return diffs.mean()

def compute_iou(masks):
    """
    Temporal IoU: IoU между соседними кадрами
    """
    masks = np.array(masks)
    T = len(masks)
    ious = []
    for t in range(1, T):
        inter = np.logical_and(masks[t-1], masks[t]).sum()
        union = np.logical_or(masks[t-1], masks[t]).sum()
        iou = inter / union if union > 0 else 1.0
        ious.append(iou)
    return np.array(ious)

def compute_area(masks):
    """
    Площадь маски в пикселях (сумма единиц)
    """
    masks = np.array(masks)
    return masks.reshape(len(masks), -1).sum(axis=1)


def apply_morphology(mask, kernel_size=5):
    """
    Морфологическая постобработка (closing)
    """
    mask_bin = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    return mask_closed.astype(np.float32)

def postprocess_mask(mask):
    # mask: float32 [0,1]
    mask = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_warped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_warped = cv2.morphologyEx(mask_warped, cv2.MORPH_OPEN, kernel)
    return mask_warped.astype(np.float32)

def save_masks(masks, out_dir, prefix="mask"):
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(masks):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        cv2.imwrite(path, (m * 255).astype("uint8"))