import cv2
import numpy as np
from io_utils import load_video_frames, load_initial_mask
from flow_utils import compute_dense_flow
from mask_utils import warp_mask, smooth_mask, save_masks, postprocess_mask

def run_vos_baseline(video_path, mask_path, beta=0.3):
    # 1. Загрузить видео и масштабировать
    frames = load_video_frames(video_path)
    mask0 = load_initial_mask(mask_path)
    H, W = mask0.shape

    # 2. Инициализация
    masks_forward = []
    masks_forward_smooth = []
    masks_postprocessed = []

    mask_prev = mask0.copy()
    mask_prev_smooth = mask0.copy()

    masks_forward.append(mask_prev)
    masks_forward_smooth.append(mask_prev_smooth)
    masks_postprocessed.append(postprocess_mask(mask_prev_smooth))

    baseline_masks = [mask0.copy() for _ in range(len(frames))]

    # 3. Forward pass
    for i in range(1, len(frames)):
        print(f"Forward: Processing frame {i}/{len(frames)-1}")
        flow = compute_dense_flow(frames[i-1], frames[i])
        mask_warped = warp_mask(mask_prev, flow)
        mask_smooth = smooth_mask(mask_prev_smooth, warp_mask(mask_prev_smooth, flow), beta)
        mask_post = postprocess_mask(mask_smooth)

        masks_forward.append(mask_warped)
        masks_forward_smooth.append(mask_smooth)
        masks_postprocessed.append(mask_post)

        mask_prev_smooth = mask_smooth
        mask_prev = mask_warped

        del flow, mask_warped, mask_smooth, mask_post

    # 4. Backward pass 
    masks_backward_smooth = [None] * len(frames)
    masks_backward_smooth[-1] = masks_forward_smooth[-1]

    for i in reversed(range(len(frames)-1)):
        print(f"Backward: Processing frame {i}/{len(frames)-1}")
        flow = compute_dense_flow(frames[i+1], frames[i])
        mask_warped = warp_mask(masks_backward_smooth[i+1], flow)
        mask_smooth = smooth_mask(masks_forward_smooth[i], mask_warped, 1 - beta)
        masks_backward_smooth[i] = mask_smooth

        del flow, mask_warped, mask_smooth
    
    out_dir = 'output'
    save_masks(baseline_masks, f"{out_dir}/baseline")
    save_masks(masks_forward, f"{out_dir}/forward")
    save_masks(masks_forward_smooth, f"{out_dir}/forward_smooth")
    save_masks(masks_postprocessed, f"{out_dir}/forward_smooth_morph")
    save_masks(masks_backward_smooth, f"{out_dir}/bidirectional")
    print('Все сохранено')

if __name__ == "__main__":
    video_path = "input_resized.mp4"   # лучше подготовить видео 640x360
    mask_path = "mask0_resized.png"    # масштабированная маска
    run_vos_baseline(video_path, mask_path)
