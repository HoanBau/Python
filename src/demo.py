import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim
import math


# 1. CHỌN ẢNH ĐẦU VÀO BẰNG FILE DIALOG
root = tk.Tk()
root.withdraw()  # Ẩn cửa sổ tkinter
image_path = filedialog.askopenfilename(
    title="Chọn ảnh đầu vào",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not image_path:
    raise FileNotFoundError("❌ Bạn chưa chọn ảnh đầu vào!")


# 2. ĐỌC ẢNH & CHUYỂN SANG ẢNH XÁM
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Không thể đọc ảnh. Vui lòng chọn lại.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. HÀM HỖ TRỢ: PSNR + ĐO THỜI GIAN
def compute_psnr(original, processed):
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


# 4. CÁC THUẬT TOÁN NGƯỠNG HÓA
results = {}

# --- Fixed Threshold ---
start_time = time.time()
_, binary_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
end_time = time.time()
results["Fixed Threshold"] = {
    "image": binary_fixed,
    "time": end_time - start_time,
    "psnr": compute_psnr(gray, binary_fixed),
    "ssim": ssim(gray, binary_fixed)
}

# --- Otsu ---
start_time = time.time()
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
end_time = time.time()
results["Otsu Method"] = {
    "image": binary_otsu,
    "time": end_time - start_time,
    "psnr": compute_psnr(gray, binary_otsu),
    "ssim": ssim(gray, binary_otsu)
}

# --- Adaptive ---
start_time = time.time()
binary_adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
end_time = time.time()
results["Adaptive Threshold"] = {
    "image": binary_adaptive,
    "time": end_time - start_time,
    "psnr": compute_psnr(gray, binary_adaptive),
    "ssim": ssim(gray, binary_adaptive)
}

# 5. HIỂN THỊ ẢNH KẾT QUẢ
titles = [
    "Ảnh gốc (RGB)",
    "Ảnh xám (Grayscale)",
    "Ngưỡng cố định (Fixed Threshold)",
    "Ngưỡng Otsu (Otsu's Method)",
    "Ngưỡng thích nghi (Adaptive Threshold)"
]
images = [image_rgb, gray, binary_fixed, binary_otsu, binary_adaptive]

plt.figure(figsize=(14, 7))
for i in range(5):
    plt.subplot(2, 3, i+1)
    cmap = "gray" if i > 0 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i], fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()

# 6. IN KẾT QUẢ SO SÁNH HIỆU SUẤT
print("\n📊 SO SÁNH HIỆU SUẤT CÁC THUẬT TOÁN:")
print("----------------------------------------------------------")
print(f"{'Thuật toán':30} {'Thời gian (s)':>15} {'PSNR':>10} {'SSIM':>10}")
print("----------------------------------------------------------")

for name, res in results.items():
    print(f"{name:30} {res['time']:.5f} {res['psnr']:.2f} {res['ssim']:.4f}")

# 7. VẼ BIỂU ĐỒ SO SÁNH
algorithms = list(results.keys())
times = [results[a]['time'] for a in algorithms]
psnrs = [results[a]['psnr'] for a in algorithms]
ssims = [results[a]['ssim'] for a in algorithms]

plt.figure(figsize=(14, 5))

# --- Biểu đồ thời gian ---
plt.subplot(1, 3, 1)
plt.bar(algorithms, times, color='skyblue')
plt.title("⏱️ Thời gian thực thi")
plt.ylabel("Thời gian (s)")
plt.xticks(rotation=15)

# --- Biểu đồ PSNR ---
plt.subplot(1, 3, 2)
plt.bar(algorithms, psnrs, color='lightgreen')
plt.title("📈 Chỉ số PSNR")
plt.ylabel("Giá trị PSNR (dB)")
plt.xticks(rotation=15)

# --- Biểu đồ SSIM ---
plt.subplot(1, 3, 3)
plt.bar(algorithms, ssims, color='salmon')
plt.title("🔍 Chỉ số SSIM")
plt.ylabel("Giá trị SSIM (0-1)")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

# 8. LƯU KẾT QUẢ RA FILE
cv2.imwrite("output_fixed.jpg", binary_fixed)
cv2.imwrite("output_otsu.jpg", binary_otsu)
cv2.imwrite("output_adaptive.jpg", binary_adaptive)

print("\n✅ Đã xử lý xong và lưu kết quả ra file ảnh!")
