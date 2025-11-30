# src/analysis.py
"""
Module phân tích kết quả threshold (Streamlit-ready):
- Tính metrics: White Ratio, Edge Preservation, Noise, SSIM, PSNR
- Tính thời gian xử lý
- Tạo và RETURN các figure (không plt.show())
- Các figure trả về có thể hiển thị bằng st.pyplot(fig)
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
    if ori_edges == 0:
        return 0.0
    return np.sum((edges_original > 0) & (edges_bw > 0)) / ori_edges

def compute_noise_ratio(bw):
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    removed_noise = np.sum(bw != opened)
    return removed_noise / bw.size

def compute_ssim(gray, bw):
    # ensure dtype and data_range
    gray_u8 = gray.astype(np.uint8)
    bw_u8 = bw.astype(np.uint8)
    try:
        return ssim(gray_u8, bw_u8, data_range=255)
    except Exception:
        # fallback: convert to float
        return ssim(gray_u8.astype(float), bw_u8.astype(float), data_range=255)

def compute_psnr(gray, bw):
    mse = np.mean((gray.astype(np.float32) - bw.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)

# ----------------------------
# LABEL HELPERS
# ----------------------------
def add_value_labels(ax, fmt="{:.3f}"):
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2,
                h, fmt.format(h),
                ha='center', va='bottom', fontsize=9)

def add_time_labels(ax):
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2,
                h, f"{h:.6f}",
                ha='center', va='bottom', fontsize=9)

# ----------------------------
# HISTOGRAM
# ----------------------------
def plot_histogram(gray, otsu_T=None, fixed_T=None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(gray.ravel(), bins=256, color='gray')
    if fixed_T is not None:
        ax.axvline(fixed_T, color='blue', linestyle='--', label=f"Fixed T={fixed_T}")
    if otsu_T is not None:
        ax.axvline(otsu_T, color='red', linestyle='-', label=f"Otsu T={otsu_T}")
    ax.set_title("Histogram & Thresholds")
    ax.legend()
    fig.tight_layout()
    return fig

# ----------------------------
# OVERLAP MATRICES
# ----------------------------
def edge_overlap_matrix(gray, bw_list, methods):
    n = len(bw_list)
    edge_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        edges_i = cv2.Canny(bw_list[i], 100, 200)
        total_i = np.sum(edges_i > 0)
        for j in range(n):
            edges_j = cv2.Canny(bw_list[j], 100, 200)
            if total_i == 0:
                edge_matrix[i, j] = 0.0
            else:
                edge_matrix[i, j] = np.sum((edges_i > 0) & (edges_j > 0)) / total_i

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(edge_matrix, annot=True, fmt=".2f",
                xticklabels=methods, yticklabels=methods, cmap="YlGnBu", ax=ax)
    ax.set_title("Edge Overlap Matrix")
    fig.tight_layout()
    return fig

def threshold_overlap_matrix(bw_list, methods):
    n = len(bw_list)
    overlap_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        white_i = np.sum(bw_list[i] == 255)
        for j in range(n):
            if white_i == 0:
                overlap_matrix[i, j] = 0.0
            else:
                overlap_matrix[i, j] = np.sum((bw_list[i] == 255) & (bw_list[j] == 255)) / white_i

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(overlap_matrix, annot=True, fmt=".2f",
                xticklabels=methods, yticklabels=methods, cmap="YlOrRd", ax=ax)
    ax.set_title("Threshold Overlap Matrix (White Pixel)")
    fig.tight_layout()
    return fig

# ----------------------------
# EXTRA PLOTS
# ----------------------------
def plot_radar_chart(methods, white, edge, noise, ssim_v, psnr_v):
    metrics = ["White", "Edge", "Noise", "SSIM", "PSNR"]
    values = np.array([white, edge, noise, ssim_v, psnr_v])  # shape (5, n_methods)

    # normalize each row to [0,1] safely
    eps = 1e-9
    maxvals = values.max(axis=1, keepdims=True)
    maxvals[maxvals == 0] = eps
    norm_vals = values / maxvals

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # close the loop

    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    for i, method in enumerate(methods):
        vals = norm_vals[:, i]
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, linewidth=2, label=method)
        ax.fill(angles, vals, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title("Radar Chart – 5 Metrics Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    return fig

def plot_ssim_psnr_scatter(methods, ssims, psnrs):
    fig, ax = plt.subplots(figsize=(6,5))
    for i, m in enumerate(methods):
        ax.scatter(ssims[i], psnrs[i], s=100)
        ax.text(ssims[i] + 0.002, psnrs[i], m, fontsize=10)
    ax.set_xlabel("SSIM")
    ax.set_ylabel("PSNR")
    ax.set_title("SSIM – PSNR Scatter Plot")
    ax.grid(True)
    fig.tight_layout()
    return fig

# ----------------------------
# LINE CHART FOR METRICS
# ----------------------------
def plot_line_compare(methods, metrics, metric_names):
    """
    Plot line charts for multiple metrics with value labels.
    metrics: 2D array, shape=(n_metrics, n_methods)
    """
    fig, ax = plt.subplots(figsize=(8,5))
    n_metrics = metrics.shape[0]
    for i, mname in enumerate(metric_names):
        ax.plot(methods, metrics[i], marker='o', label=mname)
        for j, val in enumerate(metrics[i]):
            ax.text(j, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_title("Line Comparison of Metrics")
    ax.set_xlabel("Methods")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

# ----------------------------
# METRIC MATRIX & CORR
# ----------------------------
def plot_metric_matrix(methods, metrics_data, metrics_names):
    # metrics_data shape: (n_metrics, n_methods)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(len(metrics_names)):
        sns.heatmap(metrics_data[i:i+1, :],
                    annot=True, fmt=".3f",
                    xticklabels=methods,
                    yticklabels=[metrics_names[i]],
                    cmap="coolwarm", cbar=False, ax=axes[i])
    axes[len(metrics_names)].axis("off")  # last cell off
    fig.suptitle("Metric Matrix Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_correlation_matrix(metrics_data, metrics_names):
    corr = np.corrcoef(metrics_data)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f",
                xticklabels=metrics_names, yticklabels=metrics_names,
                cmap="vlag", ax=ax)
    ax.set_title("Correlation Matrix between Metrics")
    fig.tight_layout()
    return fig

# ----------------------------
# COMPARE METHODS (MASTER)
# ----------------------------
def compare_methods(gray, bw_fixed, bw_otsu, bw_adapt,
                    fixed_T, otsu_T,
                    time_fixed, time_otsu, time_adapt):
    """
    Compute metrics and return a dict of figures:
      {
        "bar_metrics": fig,
        "histogram": fig,
        "metric_matrix": fig,
        "corr_matrix": fig,
        "edge_overlap": fig,
        "threshold_overlap": fig,
        "radar": fig,
        "ssim_psnr": fig,
        "line_metrics": fig,
        "summary": { ... }  # numeric summary if needed
      }
    """

    methods = ["Fixed", "Otsu", "Adaptive"]
    bw_list = [bw_fixed, bw_otsu, bw_adapt]

    # Compute metrics
    white_ratios = [compute_white_ratio(bw) for bw in bw_list]
    edge_preserve = [compute_edge_preservation(gray, bw) for bw in bw_list]
    noises = [compute_noise_ratio(bw) for bw in bw_list]
    ssims = [compute_ssim(gray, bw) for bw in bw_list]
    psnrs = [compute_psnr(gray, bw) for bw in bw_list]
    times = [time_fixed, time_otsu, time_adapt]

    metrics_names = ["White Ratio", "Edge", "Noise", "SSIM", "PSNR"]
    metrics_data = np.array([white_ratios, edge_preserve, noises, ssims, psnrs])

    figs = {}

    # 1) Bar chart panel (6 subplots)
    fig_bar, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    ax = axes[0]
    ax.bar(methods, white_ratios)
    ax.set_title("White Pixel Ratio")
    add_value_labels(ax)

    ax = axes[1]
    ax.bar(methods, edge_preserve)
    ax.set_title("Edge Preservation Ratio")
    add_value_labels(ax)

    ax = axes[2]
    ax.bar(methods, noises)
    ax.set_title("Noise Ratio")
    add_value_labels(ax)

    ax = axes[3]
    ax.bar(methods, ssims)
    ax.set_title("SSIM")
    add_value_labels(ax)

    ax = axes[4]
    ax.bar(methods, psnrs)
    ax.set_title("PSNR (dB)")
    add_value_labels(ax)

    ax = axes[5]
    ax.bar(methods, times)
    ax.set_title("Processing Time (s)")
    add_time_labels(ax)

    fig_bar.tight_layout()
    figs["bar_metrics"] = fig_bar

    # 2) Histogram
    figs["histogram"] = plot_histogram(gray, otsu_T=otsu_T, fixed_T=fixed_T)

    # 3) Metric matrix
    figs["metric_matrix"] = plot_metric_matrix(methods, metrics_data, metrics_names)

    # 4) Correlation matrix
    figs["corr_matrix"] = plot_correlation_matrix(metrics_data, metrics_names)

    # 5) Edge overlap
    figs["edge_overlap"] = edge_overlap_matrix(gray, bw_list, methods)

    # 6) Threshold overlap
    figs["threshold_overlap"] = threshold_overlap_matrix(bw_list, methods)

    # 7) Radar chart
    figs["radar"] = plot_radar_chart(methods, white_ratios, edge_preserve, noises, ssims, psnrs)

    # 8) SSIM-PSNR scatter
    figs["ssim_psnr"] = plot_ssim_psnr_scatter(methods, ssims, psnrs)

    # 9) Line chart of metrics
    figs["line_metrics"] = plot_line_compare(methods, metrics_data, metrics_names)

    # numeric summary (optional)
    summary = {
        "methods": methods,
        "white_ratios": white_ratios,
        "edge_preserve": edge_preserve,
        "noises": noises,
        "ssims": ssims,
        "psnrs": psnrs,
        "times": times,
        "fixed_T": fixed_T,
        "otsu_T": otsu_T
    }
    figs["summary"] = summary

    return figs
