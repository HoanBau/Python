import cv2
import numpy as np

def fixed_threshold(gray, T=127):
    """
    Ngưỡng cố định (Fixed Threshold)
    Trả về: binary image, ngưỡng T
    """
    _, bw = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return bw, T  # <- thêm T để main.py unpack được
