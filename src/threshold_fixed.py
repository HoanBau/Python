import cv2
import time

def fixed_threshold(gray, T=128):
    """
    Fixed thresholding
    Trả về: bw image, threshold value, thời gian xử lý
    """
    start = time.time()
    _, bw = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    end = time.time()
    return bw, T, end-start
