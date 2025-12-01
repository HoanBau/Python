import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

# ----------------------------
# METRICS
# ----------------------------
def compute_white_ratio(bw):
    return np.sum(bw == 255) / bw.size

def compute_edge_preservation(gray, bw):
    edges_orig = cv2.Canny(gray, 100, 200)
    edges_bw = cv2.Canny(bw, 100, 200)
    total = np.sum(edges_orig > 0)
    if total == 0:
        return 0.0
    return np.sum((edges_orig>0)&(edges_bw>0)) / total

def compute_noise_ratio(bw):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    removed_noise = np.sum(bw != opened)
    return removed_noise / bw.size

def compute_ssim(gray, bw):
    return ssim(gray.astype(np.uint8), bw.astype(np.uint8))

def compute_psnr(gray, bw):
    mse = np.mean((gray.astype(np.float32)-bw.astype(np.float32))**2)
    if mse==0:
        return float('inf')
    return 10*np.log10(255**2 / mse)

# ----------------------------
# HELPERS
# ----------------------------
def add_value_labels(ax):
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width()/2,
                patch.get_height(),
                f"{patch.get_height():.3f}",
                ha='center', va='bottom', fontsize=8)

def add_time_labels(ax):
    for patch in ax.patches:
        ax.text(patch.get_x()+patch.get_width()/2,
                patch.get_height(),
                f"{patch.get_height():.6f}",
                ha='center', va='bottom', fontsize=8)

# ----------------------------
# FIGURE FUNCTIONS
# ----------------------------
def plot_histogram(gray, otsu_T=None, fixed_T=None):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(gray.ravel(), bins=256, color='gray')
    if fixed_T is not None:
        ax.axvline(fixed_T, color='blue', linestyle='--', label=f"Fixed T={fixed_T}")
    if otsu_T is not None:
        ax.axvline(otsu_T, color='red', linestyle='-', label=f"Otsu T={otsu_T}")
    ax.set_title("Histogram & Thresholds", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_line_compare(methods, metrics, metric_names):
    fig, ax = plt.subplots(figsize=(6,3))
    for i, name in enumerate(metric_names):
        ax.plot(methods, metrics[i], marker='o', label=name)
        for j, val in enumerate(metrics[i]):
            ax.text(j, val, f"{val:.3f}", ha='center', va='bottom', fontsize=8)
    ax.set_title("Line Comparison of Metrics", fontsize=9)
    ax.set_xlabel("Methods")
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    return fig

def edge_overlap_matrix(bw_list, methods):
    n = len(bw_list)
    mat = np.zeros((n,n))
    for i in range(n):
        edges_i = cv2.Canny(bw_list[i],100,200)
        total_i = np.sum(edges_i>0)
        for j in range(n):
            edges_j = cv2.Canny(bw_list[j],100,200)
            mat[i,j] = 0.0 if total_i==0 else np.sum((edges_i>0)&(edges_j>0))/total_i
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=methods, yticklabels=methods, cmap="YlGnBu", ax=ax)
    ax.set_title("Edge Overlap Matrix", fontsize=9)
    fig.tight_layout()
    return fig

def threshold_overlap_matrix(bw_list, methods):
    n = len(bw_list)
    mat = np.zeros((n,n))
    for i in range(n):
        white_i = np.sum(bw_list[i]==255)
        for j in range(n):
            mat[i,j] = 0.0 if white_i==0 else np.sum((bw_list[i]==255)&(bw_list[j]==255))/white_i
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=methods, yticklabels=methods, cmap="YlOrRd", ax=ax)
    ax.set_title("Threshold Overlap Matrix", fontsize=9)
    fig.tight_layout()
    return fig

def plot_radar_chart(methods, white, edge, noise, ssims, psnrs):
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = ["White","Edge","Noise","SSIM","PSNR"]
    values = np.array([white, edge, noise, ssims, psnrs])
    # normalize
    values_norm = values/values.max(axis=1, keepdims=True)
    angles = np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    for i, method in enumerate(methods):
        v = values_norm[:,i].tolist()
        v += v[:1]
        ax.plot(angles, v, label=method)
        ax.fill(angles, v, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title("Radar Chart – 5 Metrics Comparison", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=8)
    fig.tight_layout()
    return fig

def plot_ssim_psnr_scatter(methods, ssims, psnrs):
    fig, ax = plt.subplots(figsize=(5,3.5))
    for i, m in enumerate(methods):
        ax.scatter(ssims[i], psnrs[i], s=60)
        ax.text(ssims[i]+0.002, psnrs[i], m, fontsize=8)
    ax.set_xlabel("SSIM", fontsize=9)
    ax.set_ylabel("PSNR", fontsize=9)
    ax.set_title("SSIM–PSNR Scatter Plot", fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_metric_matrix(methods, metrics_data, metrics_names):
    fig, axes = plt.subplots(2,3,figsize=(6,4))
    axes = axes.flatten()
    for i in range(len(metrics_names)):
        sns.heatmap(metrics_data[i:i+1,:], annot=True, fmt=".3f",
                    xticklabels=methods, yticklabels=[metrics_names[i]],
                    cmap="coolwarm", cbar=False, ax=axes[i])
        axes[i].tick_params(axis='both', labelsize=8)
    axes[len(metrics_names)].axis("off")
    fig.suptitle("Metric Matrix Comparison", fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.95])
    return fig

def plot_correlation_matrix(metrics_data, metrics_names):
    corr = np.corrcoef(metrics_data)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=metrics_names, yticklabels=metrics_names, cmap="vlag", ax=ax)
    ax.set_title("Correlation Matrix between Metrics", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    return fig

# ----------------------------
# MASTER FUNCTION
# ----------------------------
def compare_methods(gray, bw_fixed, bw_otsu, bw_adapt,
                    fixed_T, otsu_T,
                    time_fixed, time_otsu, time_adapt):

    methods = ["Fixed","Otsu","Adaptive"]
    bw_list = [bw_fixed, bw_otsu, bw_adapt]

    white_ratios = [compute_white_ratio(bw) for bw in bw_list]
    edge_preserve = [compute_edge_preservation(gray,bw) for bw in bw_list]
    noises = [compute_noise_ratio(bw) for bw in bw_list]
    ssims = [compute_ssim(gray,bw) for bw in bw_list]
    psnrs = [compute_psnr(gray,bw) for bw in bw_list]
    times = [time_fixed,time_otsu,time_adapt]

    metrics_data = np.array([white_ratios, edge_preserve, noises, ssims, psnrs])
    metrics_names = ["White Ratio","Edge","Noise","SSIM","PSNR"]

    figs = {}
    # bar metrics
    fig_bar, axes = plt.subplots(3,2,figsize=(10,8))
    axes = axes.flatten()
    for idx, (ax, title, values) in enumerate(zip(
        axes,
        ["White Pixel Ratio","Edge Preservation Ratio","Noise Ratio","SSIM","PSNR","Processing Time (s)"],
        [white_ratios, edge_preserve, noises, ssims, psnrs, times])):
        ax.bar(methods, values)
        ax.set_title(title, fontsize=8)
        add_value_labels(ax) if idx!=5 else add_time_labels(ax)
    fig_bar.tight_layout()
    figs["bar_metrics"] = fig_bar

    figs["histogram"] = plot_histogram(gray, otsu_T, fixed_T)
    figs["metric_matrix"] = plot_metric_matrix(methods, metrics_data, metrics_names)
    figs["corr_matrix"] = plot_correlation_matrix(metrics_data, metrics_names)
    figs["edge_overlap"] = edge_overlap_matrix(bw_list, methods)
    figs["threshold_overlap"] = threshold_overlap_matrix(bw_list, methods)
    figs["radar"] = plot_radar_chart(methods, white_ratios, edge_preserve, noises, ssims, psnrs)
    figs["ssim_psnr"] = plot_ssim_psnr_scatter(methods, ssims, psnrs)
    figs["line_metrics"] = plot_line_compare(methods, metrics_data, metrics_names)
    figs["summary"] = {
        "methods":methods,"white_ratios":white_ratios,"edge":edge_preserve,
        "noises":noises,"ssims":ssims,"psnrs":psnrs,"times":times,
        "fixed_T":fixed_T,"otsu_T":otsu_T
    }

    return figs
