import streamlit as st
import cv2
import numpy as np

from src.threshold_fixed import fixed_threshold
from src.threshold_otsu import otsu_threshold
from src.threshold_adaptive import adaptive_threshold
from src.analysis import compare_methods

st.set_page_config(page_title="Image Thresholding App", layout="wide")
st.title("🖼️ Ứng dụng xử lý ảnh - So sánh 3 phương pháp Threshold")

uploaded_file = st.file_uploader("Tải ảnh màu lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Ảnh gốc")
    st.image(img, channels="BGR", width=400)

    # 2. Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Áp dụng 3 phương pháp Threshold
    bw_fixed, T_fixed = fixed_threshold(gray)
    bw_otsu, T_otsu = otsu_threshold(gray)
    bw_adapt = adaptive_threshold(gray)

    st.subheader("🔳 Kết quả 3 phương pháp")
    col1, col2, col3 = st.columns(3)
    col1.image(bw_fixed, caption=f"Fixed Threshold (T={T_fixed})", width=300)
    col2.image(bw_adapt, caption="Adaptive Threshold", width=300)
    col3.image(bw_otsu, caption=f"Otsu Threshold (T={T_otsu})", width=300)

    # 4. Phân tích & vẽ biểu đồ
    st.subheader("📊 Biểu đồ phân tích metrics & ma trận")
    figs = compare_methods(gray, bw_fixed, bw_otsu, bw_adapt, T_fixed, T_otsu, 0,0,0)

    # Hiển thị tất cả figure
    for name, fig in figs.items():
        if name == "summary":
            continue
        st.subheader(name.replace("_"," ").title())
        st.pyplot(fig)

else:
    st.info("Hãy upload một ảnh để bắt đầu.")
