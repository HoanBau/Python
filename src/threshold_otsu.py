import cv2
import time

def otsu_threshold(gray):
    """
    Otsu thresholding
    Trả về: bw image, threshold value, thời gian xử lý
    """
    start = time.time()
    T, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    end = time.time()
    return bw, T, end-start
