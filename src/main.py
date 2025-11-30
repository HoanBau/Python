# main.py
"""
Chương trình chính:
- Đọc ảnh
- Tiền xử lý
- Áp dụng 3 kỹ thuật threshold
- Đo thời gian xử lý
- Phân tích & so sánh bằng biểu đồ: White Ratio, Edge Preservation, Noise, SSIM, PSNR, Time, Histogram
- Vẽ 4 ma trận:
    1. Metric Matrix
    2. Correlation Matrix
    3. Edge Overlap Matrix
    4. Threshold Overlap Matrix
- Lưu ảnh output
"""

from utils import read_image, save_image, ensure_dir, show_images
from preprocess import to_gray, gaussian_blur
from threshold_fixed import fixed_threshold
from threshold_otsu import otsu_threshold
from threshold_adaptive import adaptive_threshold
from analysis import compare_methods
import time

def process_image(img_path):
    """Hàm xử lý ảnh đầy đủ"""
    print(f"Đang xử lý ảnh: {img_path}")

    # -------------------------
    # 1. Đọc ảnh màu
    # -------------------------
    img = read_image(img_path)

    # -------------------------
    # 2. Chuyển sang grayscale
    # -------------------------
    gray = to_gray(img)

    # -------------------------
    # 3. Làm mờ để giảm nhiễu
    # -------------------------
    blur = gaussian_blur(gray)

    # -------------------------
    # 4. Áp dụng 3 phương pháp Threshold + đo thời gian
    # -------------------------
    # --- Fixed Threshold ---
    start = time.time()
    bw_fixed, fixed_T = fixed_threshold(blur, 127)  # trả về ảnh nhị phân và ngưỡng
    time_fixed = time.time() - start

    # --- Otsu Threshold ---
    start = time.time()
    bw_otsu, otsu_T = otsu_threshold(blur)
    time_otsu = time.time() - start

    # --- Adaptive Threshold ---
    start = time.time()
    bw_adapt = adaptive_threshold(blur, blockSize=15, C=2)
    time_adapt = time.time() - start

    # -------------------------
    # 5. Lưu ảnh kết quả
    # -------------------------
    ensure_dir("outputs/fixed")
    ensure_dir("outputs/otsu")
    ensure_dir("outputs/adaptive")

    save_image(bw_fixed, "outputs/fixed/result_fixed.png")
    save_image(bw_otsu,  "outputs/otsu/result_otsu.png")
    save_image(bw_adapt, "outputs/adaptive/result_adaptive.png")

    print("Đã lưu ảnh vào folder outputs/ ✓")

    # -------------------------
    # 6. Hiển thị ảnh để so sánh trực quan
    # -------------------------
    show_images(
        [gray, bw_fixed, bw_otsu, bw_adapt],
        ["Grayscale", "Fixed Threshold", "Otsu", "Adaptive Threshold"]
    )

    # -------------------------
    # 7. Phân tích & vẽ biểu đồ metrics + 4 ma trận
    # -------------------------
    compare_methods(
        gray,
        bw_fixed, bw_otsu, bw_adapt,
        fixed_T, otsu_T,
        time_fixed, time_otsu, time_adapt
    )

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Chọn ảnh input
    process_image("data/input.jpg")
