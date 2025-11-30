import cv2
import numpy as np


def otsu_threshold(gray):
    """
    Ngưỡng Otsu
    Trả về: binary image, ngưỡng T
    """
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    T, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[Otsu] Ngưỡng tối ưu = {T}")
    return bw, T  # <- thêm T để main.py unpack được
