import streamlit as st
import cv2
import numpy as np

from src.threshold_fixed import fixed_threshold
from src.threshold_adaptive import adaptive_threshold
from src.threshold_otsu import otsu_threshold
from src.analysis import plot_histogram, plot_line_compare, compare_methods

st.set_page_config(page_title="Image Thresholding App", layout="wide")

st.title("🖼️ Ứng dụng xử lý ảnh - So sánh 3 phương pháp Threshold")

uploaded_file = st.file_uploader("Tải ảnh màu lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Ảnh gốc:")
    st.image(img, channels="BGR", width=450)

    # chuyển sang Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # chạy 3 thuật toán
    bw_fixed, T_fixed, t_fixed = fixed_threshold(gray)    # trả thêm thời gian
    bw_adp, t_adapt = adaptive_threshold(gray)            # trả thêm thời gian
    bw_otsu, T_otsu, t_otsu = otsu_threshold(gray)        # trả thêm thời gian

    st.subheader("🔳 Kết quả 3 phương pháp")
    col1, col2, col3 = st.columns(3)
    col1.image(bw_fixed, caption=f"Fixed Threshold (T={T_fixed})", width=300)
    col2.image(bw_adp, caption=f"Adaptive Threshold", width=300)
    col3.image(bw_otsu, caption=f"Otsu Threshold (T={T_otsu})", width=300)

    # Histogram
    st.subheader("📊 Histogram mức xám")
    fig_hist = plot_histogram(gray, fixed_T=T_fixed, otsu_T=T_otsu)
    st.pyplot(fig_hist)

    # Biểu đồ trung bình pixel
    st.subheader("📈 Biểu đồ minh họa trung bình pixel")
    methods = ["Fixed", "Adaptive", "Otsu"]
    metrics = np.array([[np.mean(bw_fixed), np.mean(bw_adp), np.mean(bw_otsu)]])
    metric_names = ["Mean Pixel Value"]
    fig_line = plot_line_compare(methods, metrics, metric_names)
    st.pyplot(fig_line)

    # So sánh đầy đủ các metrics
    st.subheader("📊 So sánh đầy đủ các metrics")
    figs = compare_methods(gray, bw_fixed, bw_otsu, bw_adp,
                           T_fixed, T_otsu, t_fixed, t_otsu, t_adapt)
    
    for key, fig in figs.items():
        if key == "summary":
            st.subheader("🔹 Numeric summary")
            st.table(fig)  # hoặc st.json(fig)
        else:
            st.subheader(key.replace("_", " ").title())
            st.pyplot(fig)

else:
    st.info("Hãy upload một ảnh để bắt đầu.")
