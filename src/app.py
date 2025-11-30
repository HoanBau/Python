import streamlit as st
import cv2
import numpy as np

from src.threshold_fixed import apply_fixed_threshold
from src.threshold_adaptive import apply_adaptive_threshold
from src.threshold_otsu import apply_otsu_threshold
from src.analysis import plot_histogram, plot_line_compare

st.set_page_config(page_title="Image Thresholding App", layout="wide")

st.title("🖼️ Ứng dụng xử lý ảnh - So sánh 3 phương pháp Threshold")

uploaded_file = st.file_uploader("Tải ảnh màu lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Ảnh gốc:")
    st.image(img, channels="BGR", width=450)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th_fixed = apply_fixed_threshold(gray)
    th_adp = apply_adaptive_threshold(gray)
    th_otsu = apply_otsu_threshold(gray)

    st.subheader("🔳 Ảnh trắng đen theo 3 phương pháp")
    col1, col2, col3 = st.columns(3)
    col1.image(th_fixed, caption="Fixed Threshold", width=300)
    col2.image(th_adp, caption="Adaptive Threshold", width=300)
    col3.image(th_otsu, caption="Otsu Threshold", width=300)

    st.subheader("📊 Histogram mức xám")
    st.pyplot(plot_histogram(gray))

    st.subheader("📈 Biểu đồ so sánh mẫu")
    sample = {
        "Fixed": np.mean(th_fixed),
        "Adaptive": np.mean(th_adp),
        "Otsu": np.mean(th_otsu)
    }
    st.pyplot(plot_line_compare(sample))

else:
    st.info("Hãy upload một ảnh để bắt đầu.")
