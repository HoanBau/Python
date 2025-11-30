"""
Module phân tích kết quả threshold:
- Tính metrics: White Ratio, Edge Preservation, Noise, SSIM, PSNR
- Tính thời gian xử lý
- Vẽ biểu đồ so sánh (có hiển thị số trên từng cột)
- Vẽ 4 ma trận:
    1. Metric Matrix
    2. Correlation Matrix giữa metrics
    3. Edge Overlap Matrix
    4. Threshold Overlap Matrix (pixel trùng nhau)
- Thêm 2 biểu đồ nâng cao:
    5. Radar Chart (Tổng hợp 5 metrics)
    6. Scatter SSIM–PSNR
- Line chart nâng cao: plot_line_compare
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

# ----------------------------
# BASIC METRICS
# ----------------------------
def compute_white_ratio(bw):
    return np.sum(bw == 255) / bw.size

def compute_edge_preservation(gray, bw):
    edges_original = cv2.Canny(gray, 100, 200)
    edges_bw = cv2.Canny(bw, 100, 200)
    ori_edges = np.sum(edges_original > 0)
    bw_edges = np.sum(edges_bw > 0)
    if ori_edges == 0:
        return 0
    return bw_edges / ori_edges

def compute_noise_ratio(bw):
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    removed_noise = np.sum(bw != opened)
    return removed_noise / bw.size

def compute_ssim(gray, bw):
    bw_norm = bw.astype(np.uint8)
    return ssim(gray, bw_norm)

def compute_psnr(gray, bw):
    mse = np.mean((gray.astype(np.float32) - bw.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

# ----------------------------
# HISTOGRAM
# ----------------------------
def plot_histogram(gray, otsu_T=None, fixed_T=None):
    plt.figure(figsize=(8,4))
    plt.hist(gray.ravel(), bins=256, color='gray')
    if fixed_T is not None:
        plt.axvline(fixed_T, color='blue', linestyle='--', label=f"Fixed T={fixed_T}")
    if otsu_T is not None:
        plt.axvline(otsu_T, color='red', linestyle='-', label=f"Otsu T={otsu_T}")
    plt.title("Histogram & Thresholds")
    plt.legend()
    plt.show()

# ----------------------------
# OVERLAP MATRICES
# ----------------------------
def edge_overlap_matrix(gray, bw_list, methods):
    n = len(bw_list)
    edge_matrix = np.zeros((n,n))
    for i in range(n):
        edges_i = cv2.Canny(bw_list[i], 100, 200)
        for j in range(n):
            edges_j = cv2.Canny(bw_list[j], 100, 200)
            if np.sum(edges_i > 0) == 0:
                edge_matrix[i,j] = 0
            else:
                edge_matrix[i,j] = np.sum((edges_i>0) & (edges_j>0)) / np.sum(edges_i>0)

    plt.figure(figsize=(6,5))
    sns.heatmap(edge_matrix, annot=True, fmt=".2f",
                xticklabels=methods, yticklabels=methods, cmap="YlGnBu")
    plt.title("Edge Overlap Matrix")
    plt.show()


def threshold_overlap_matrix(bw_list, methods):
    n = len(bw_list)
    overlap_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            white_i = np.sum(bw_list[i] == 255)
            overlap_matrix[i,j] = np.sum((bw_list[i]==255)&(bw_list[j]==255)) / white_i

    plt.figure(figsize=(6,5))
    sns.heatmap(overlap_matrix, annot=True, fmt=".2f",
                xticklabels=methods, yticklabels=methods, cmap="YlOrRd")
    plt.title("Threshold Overlap Matrix (White Pixel)")
    plt.show()

# ----------------------------
# LABEL FUNCTIONS
# ----------------------------
def add_value_labels(ax):
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2,
                h, f"{h:.3f}",
                ha='center', va='bottom', fontsize=9)

def add_time_labels(ax):
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2,
                h, f"{h:.6f}",
                ha='center', va='bottom', fontsize=9)

# ----------------------------
# EXTRA PLOTS
# ----------------------------
def plot_radar_chart(methods, white, edge, noise, ssim_v, psnr_v):
    metrics = ["White", "Edge", "Noise", "SSIM", "PSNR"]
    values = np.array([white, edge, noise, ssim_v, psnr_v])

    # Normalize values to 0-1 for radar plot
    values_norm = []
    for v in values:
        vmax = max(v)
        if vmax==0:
            values_norm.append(v)
        else:
            values_norm.append(v / vmax)
    values_norm = np.array(values_norm)

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    for i, method in enumerate(methods):
        vals = values_norm[:,i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=method)
        ax.fill(angles, vals, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    plt.title("Radar Chart – 5 Metrics Comparison")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.show()

def plot_ssim_psnr_scatter(methods, ssims, psnrs):
    plt.figure(figsize=(6,5))
    for i, m in enumerate(methods):
        plt.scatter(ssims[i], psnrs[i], s=100)
        plt.text(ssims[i] + 0.002, psnrs[i], m, fontsize=10)
    plt.xlabel("SSIM")
    plt.ylabel("PSNR")
    plt.title("SSIM – PSNR Scatter Plot")
    plt.grid(True)
    plt.show()

# ----------------------------
# LINE CHART FOR METRICS (NEW)
# ----------------------------
def plot_line_compare(methods, metrics, metric_names):
    """
    Plot line charts for multiple metrics with value labels.
    metrics: 2D array, shape=(n_metrics, n_methods)
    """
    plt.figure(figsize=(8,5))
    for i, mname in enumerate(metric_names):
        plt.plot(methods, metrics[i], marker='o', label=mname)
        for j, val in enumerate(metrics[i]):
            plt.text(j, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    plt.title("Line Comparison of Metrics")
    plt.xlabel("Methods")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# COMPARE METHODS
# ----------------------------
def compare_methods(gray, bw_fixed, bw_otsu, bw_adapt,
                    fixed_T, otsu_T,
                    time_fixed, time_otsu, time_adapt):

    methods = ["Fixed", "Otsu", "Adaptive"]
    bw_list = [bw_fixed, bw_otsu, bw_adapt]

    # Compute metrics
    white_ratios = [compute_white_ratio(bw) for bw in bw_list]
    edge_preserve = [compute_edge_preservation(gray, bw) for bw in bw_list]
    noises = [compute_noise_ratio(bw) for bw in bw_list]
    ssims = [compute_ssim(gray, bw) for bw in bw_list]
    psnrs = [compute_psnr(gray, bw) for bw in bw_list]
    times = [time_fixed, time_otsu, time_adapt]

    # ----------------------------
    # 1. BAR CHART METRICS
    # ----------------------------
    plt.figure(figsize=(12,10))

    ax = plt.subplot(3,2,1)
    ax.bar(methods, white_ratios, color="#4CAF50")
    plt.title("White Pixel Ratio")
    add_value_labels(ax)

    ax = plt.subplot(3,2,2)
    ax.bar(methods, edge_preserve, color="#2196F3")
    plt.title("Edge Preservation Ratio")
    add_value_labels(ax)

    ax = plt.subplot(3,2,3)
    ax.bar(methods, noises, color="#FF9800")
    plt.title("Noise Ratio")
    add_value_labels(ax)

    ax = plt.subplot(3,2,4)
    ax.bar(methods, ssims, color="#9C27B0")
    plt.title("SSIM")
    add_value_labels(ax)

    ax = plt.subplot(3,2,5)
    ax.bar(methods, psnrs, color="#F44336")
    plt.title("PSNR (dB)")
    add_value_labels(ax)

    ax = plt.subplot(3,2,6)
    ax.bar(methods, times, color="#795548")
    plt.title("Processing Time (s)")
    add_time_labels(ax)

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 2. HISTOGRAM
    # ----------------------------
    plot_histogram(gray, otsu_T, fixed_T)

    # ----------------------------
    # 3. METRIC MATRIX
    # ----------------------------
    metrics_names = ["White Ratio", "Edge", "Noise", "SSIM", "PSNR"]
    metrics_data = np.array([white_ratios, edge_preserve, noises, ssims, psnrs])

    fig, axes = plt.subplots(2,3,figsize=(12,8))
    axes = axes.flatten()
    for i in range(5):
        sns.heatmap(metrics_data[i:i+1,:],
                    annot=True, fmt=".3f",
                    xticklabels=methods,
                    yticklabels=[metrics_names[i]],
                    cmap="coolwarm", cbar=False, ax=axes[i])
    axes[5].axis("off")
    plt.suptitle("Metric Matrix Comparison")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 4. CORRELATION MATRIX
    # ----------------------------
    metrics_arr = np.array([white_ratios, edge_preserve, noises, ssims, psnrs])
    corr = np.corrcoef(metrics_arr)

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f",
                xticklabels=metrics_names, yticklabels=metrics_names,
                cmap="vlag")
    plt.title("Correlation Matrix between Metrics")
    plt.show()

    # ----------------------------
    # 5. EDGE OVERLAP MATRIX
    # ----------------------------
    edge_overlap_matrix(gray, bw_list, methods)

    # ----------------------------
    # 6. THRESHOLD OVERLAP MATRIX
    # ----------------------------
    threshold_overlap_matrix(bw_list, methods)

    # ----------------------------
    # 7. RADAR CHART
    # ----------------------------
    plot_radar_chart(methods, white_ratios, edge_preserve,
                     noises, ssims, psnrs)

    # ----------------------------
    # 8. SSIM–PSNR SCATTER PLOT
    # ----------------------------
    plot_ssim_psnr_scatter(methods, ssims, psnrs)

    # ----------------------------
    # 9. LINE CHART METRICS
    # ----------------------------
    plot_line_compare(methods, metrics_data, metrics_names)
