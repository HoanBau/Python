# preprocess.py
"""
Chứa các hàm tiền xử lý ảnh:
- Chuyển sang grayscale
- Resize ảnh
- Làm mờ Gaussian
- Tăng cường tương phản CLAHE
"""

import cv2

def to_gray(img_bgr):
    """Chuyển ảnh BGR sang grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_keep_ratio(img, width=None, height=None):
    """
    Resize ảnh giữ tỉ lệ.
    Chỉ cần truyền width HOẶC height.
    """
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    
    if width is not None:
        scale = width / w
        new_w, new_h = width, int(h * scale)
    else:
        scale = height / h
        new_h, new_w = height, int(w * scale)

    return cv2.resize(img, (new_w, new_h))

def gaussian_blur(gray, ksize=(5,5)):
    """Làm mờ Gaussian để giảm nhiễu."""
    return cv2.GaussianBlur(gray, ksize, 0)

def clahe(gray):
    """Tăng cường tương phản ảnh bằng CLAHE."""
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return c.apply(gray)
