import cv2
import os
import matplotlib.pyplot as plt

# ----------------------------
# Đọc ảnh từ file
# ----------------------------
def read_image(path):
    return cv2.imread(path)

# ----------------------------
# Lưu ảnh ra file
# ----------------------------
def save_image(img,path):
    cv2.imwrite(path,img)

# ----------------------------
# Tạo folder nếu chưa tồn tại
# ----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------
# Hiển thị nhiều ảnh cùng lúc
# ----------------------------
def show_images(img_list, titles):
    n = len(img_list)
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        if len(img_list[i].shape)==2:
            plt.imshow(img_list[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
