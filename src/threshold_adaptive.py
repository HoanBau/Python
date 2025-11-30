# threshold_adaptive.py
"""
Ngưỡng thích nghi (Adaptive Threshold)
"""

import cv2  # quan trọng
import numpy as np

def adaptive_threshold(gray, blockSize=15, C=2, method="gaussian"):
    """
    Ngưỡng thích nghi chia ảnh thành vùng nhỏ (blockSize).
    - blockSize: kích thước vùng (phải là số lẻ, ví dụ 11, 15, 21)
    - C: hằng số trừ đi
    - method: 'mean' hoặc 'gaussian'
    """
    m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" \
        else cv2.ADAPTIVE_THRESH_MEAN_C

    binary = cv2.adaptiveThreshold(
        gray, 255,
        m,
        cv2.THRESH_BINARY,
        blockSize,
        C
    )
    return binary  # hoặc return binary, blockSize nếu muốn
