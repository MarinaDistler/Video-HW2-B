import cv2
import numpy as np

def load_video_frames(video_path):
    """
    Необходимо реализовать загрузку всех кадров из видео.

    Требования:
    - Открыть видео по пути video_path.
    - Последовательно прочитать все кадры до конца файла.
    - Сохранить кадры в список (в формате numpy-массивов).
    - Вернуть список кадров.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Конвертируем в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def load_initial_mask(mask_path):
    """
    Загрузка маски первого кадра.
    - Загружаем изображение маски в градациях серого.
    - Приводим значения к диапазону [0, 1].
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = mask.astype(np.float32) / 255.0
    return mask
