import cv2
import time

def adaptive_threshold(gray, block_size=11, C=2):
    """
    Adaptive thresholding
    Trả về: bw image, thời gian xử lý
    """
    start = time.time()
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, C)
    end = time.time()
    return bw, end-start
