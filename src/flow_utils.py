import cv2
import numpy as np

def compute_dense_flow(frame_prev, frame_next):
    """
    Расчёт плотного оптического потока между двумя кадрами.
    Используется Farnebäck.
    """
    prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=10,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow
