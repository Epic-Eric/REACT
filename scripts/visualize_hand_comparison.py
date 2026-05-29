#!/usr/bin/env python3
"""
Generate side-by-side hand pose comparison figures for the final report.

Left panel:  Ground truth hand skeleton (keypoints + connections) — what a
             camera-based hand tracker produces.
Right panel: Predicted 3D hand mesh reconstruction — the model output.

Usage:
    python scripts/visualize_hand_comparison.py \
        --hdf5 /path/to/session.hdf5 \
        --checkpoint /path/to/checkpoint.ckpt \
        --output paper/final_report/hand_comparison.png

    # Or just generate from ground truth data (no model needed):
    python scripts/visualize_hand_comparison.py \
        --hdf5 /path/to/session.hdf5 \
        --output paper/final_report/hand_gt_only.png \
        --gt-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Add project roots to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "emg2pose"))
sys.path.insert(0, str(PROJECT_ROOT / "emg2pose" / "emg2pose" / "UmeTrack"))

from emg2pose.data import Emg2PoseSessionData
from emg2pose.kinematics import forward_kinematics, load_default_hand_model
from emg2pose.visualization import skin_mesh_from_angles
from emg2pose.utils import downsample

# ── Hand skeleton topology ────────────────────────────────────────────────
# 21 landmarks: 5 fingertips (0-4), wrist (5), intermediate joints (6-20), palm (20)
# Connections per finger (wrist → proximal → intermediate → distal → tip)
FINGER_CONNECTIONS = {
    "Thumb":  [(5, 6), (6, 7), (7, 0)],
    "Index":  [(5, 8), (8, 9), (9, 10), (10, 1)],
    "Middle": [(5, 11), (11, 12), (12, 13), (13, 2)],
    "Ring":   [(5, 14), (14, 15), (15, 16), (16, 3)],
    "Pinky":  [(5, 17), (17, 18), (18, 19), (19, 4)],
}
PALM_CONNECTIONS = [(5, 20)]  # wrist → palm center

FINGER_COLORS = {
    "Thumb":  "#FF6B6B",
    "Index":  "#4ECDC4",
    "Middle": "#45B7D1",
    "Ring":   "#FFA07A",
    "Pinky":  "#DDA0DD",
}
PALM_COLOR = "#888888"

# ── Poster aesthetic palette ──────────────────────────────────────────────
# Soft, friendly pastels keyed to the deep blue #1132AB theme accent.
POSTER_ACCENT      = "#1132AB"   # theme color (titles, wrist marker, EMG colormap)
POSTER_ACCENT_SOFT = "#7A8AAB"   # desaturated for inner joints / palm
POSTER_FINGERTIP   = "#FF8FA3"   # warm rose for fingertip dots
POSTER_GT_MESH     = "#A8D8EA"   # pastel sky — ground truth
POSTER_PRED_MESH   = "#FFC4B5"   # pastel peach — REACT prediction
POSTER_TEXT        = "#000000"   # plain black for all labels / titles

POSTER_FINGER_COLORS = {
    "Thumb":  "#F4A6B8",   # blush
    "Index":  "#9AD2C3",   # mint
    "Middle": "#A8C5E9",   # baby blue
    "Ring":   "#F6C28C",   # peach
    "Pinky":  "#C8A2D8",   # lilac
}
POSTER_PALM_COLOR = "#B8C4D9"


def _setup_poster_style():
    """Set matplotlib rcParams for an elegant, rounded poster look."""
    plt.rcParams.update({
        "font.family":      ["Montserrat",
                             "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.weight":      "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor":   "#CCCCCC",
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "xtick.color":      "#666666",
        "ytick.color":      "#666666",
        "text.color":       POSTER_TEXT,
        "savefig.facecolor": "white",
    })


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing along the last axis."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="same"), -1, x
    )


def joint_angles_to_landmarks(joint_angles: np.ndarray) -> np.ndarray:
    """Convert joint angles (20,) to 3D landmark positions (21, 3).

    Pads 2 null wrist DOFs to get 22 DOFs required by forward kinematics.
    """
    ja = torch.from_numpy(joint_angles).float()
    # FK expects (B, C, T) — treat single frame as (1, 20, 1)
    ja = ja.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
    landmarks = forward_kinematics(ja)   # (1, 1, 21, 3)
    return landmarks[0, 0].numpy()        # (21, 3)


def plot_skeleton(ax, landmarks: np.ndarray, title: str = "Ground Truth Keypoints"):
    """Plot hand skeleton as keypoints + colored finger connections on a 3D axis."""
    # Draw finger connections
    for finger, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for i, j in connections:
            xs = [landmarks[i, 0], landmarks[j, 0]]
            ys = [landmarks[i, 1], landmarks[j, 1]]
            zs = [landmarks[i, 2], landmarks[j, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2.5, solid_capstyle="round")

    # Palm connection
    for i, j in PALM_CONNECTIONS:
        xs = [landmarks[i, 0], landmarks[j, 0]]
        ys = [landmarks[i, 1], landmarks[j, 1]]
        zs = [landmarks[i, 2], landmarks[j, 2]]
        ax.plot(xs, ys, zs, color=PALM_COLOR, linewidth=2, linestyle="--")

    # Draw keypoints: fingertips larger, others smaller
    fingertip_idx = [0, 1, 2, 3, 4]
    wrist_idx = [5]
    other_idx = list(range(6, 21))

    ax.scatter(*landmarks[fingertip_idx].T, s=80, c="#FF4444", zorder=5,
               edgecolors="white", linewidths=0.8, label="Fingertips")
    ax.scatter(*landmarks[wrist_idx].T, s=100, c="#333333", zorder=5,
               marker="s", edgecolors="white", linewidths=0.8, label="Wrist")
    ax.scatter(*landmarks[other_idx].T, s=40, c="#666666", zorder=5,
               edgecolors="white", linewidths=0.5, label="Joints")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.legend(fontsize=11, loc="upper left")


def plot_mesh(ax, joint_angles: np.ndarray, title: str = "Predicted Reconstruction",
              color: str = "lightpink"):
    """Plot hand mesh on a 3D axis."""
    vertices, triangles = skin_mesh_from_angles(joint_angles)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color=color, alpha=1.0, edgecolor="none", linewidth=0.0,
        shade=True, antialiased=True,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)


def style_3d_axis(ax, landmarks: np.ndarray):
    """Apply consistent styling to a 3D axis."""
    # Auto-range from landmarks
    margin = 15
    for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        lo, hi = landmarks[:, dim].min() - margin, landmarks[:, dim].max() + margin
        setter(lo, hi)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.2)

    # Camera angle
    ax.view_init(elev=15, azim=-60)


def generate_comparison_figure(
    gt_joint_angles: np.ndarray,
    pred_joint_angles: np.ndarray | None = None,
    frame_idx: int = 0,
    save_path: str | Path | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate a side-by-side comparison figure.

    Args:
        gt_joint_angles: Ground truth (T, 20) at 30Hz.
        pred_joint_angles: Predictions (T, 20) at 30Hz. If None, shows GT mesh on right.
        frame_idx: Which frame to visualize.
        save_path: Path to save the figure.
        dpi: Output DPI.

    Returns:
        Matplotlib Figure.
    """
    gt_ja = gt_joint_angles[frame_idx]
    landmarks_gt = joint_angles_to_landmarks(gt_ja)

    if pred_joint_angles is not None:
        pred_ja = pred_joint_angles[frame_idx]
        landmarks_pred = joint_angles_to_landmarks(pred_ja)
    else:
        pred_ja = gt_ja
        landmarks_pred = landmarks_gt

    fig = plt.figure(figsize=(14, 6), facecolor="white")

    # Left: skeleton keypoints
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_skeleton(ax1, landmarks_gt, title="Hand Tracking Keypoints\n(Ground Truth)")
    style_3d_axis(ax1, landmarks_gt)

    # Right: mesh reconstruction
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    right_title = "3D Mesh Reconstruction\n(Model Prediction)" if pred_joint_angles is not None else "3D Mesh Reconstruction\n(Ground Truth)"
    right_color = "lightpink" if pred_joint_angles is not None else "lightblue"
    plot_mesh(ax2, pred_ja, title=right_title, color=right_color)
    style_3d_axis(ax2, landmarks_pred)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")

    return fig


# ── Poster-aesthetic plotting helpers ─────────────────────────────────────

def style_3d_axis_poster(ax, landmarks: np.ndarray, margin: float = 6.0):
    """Clean, axis-free 3D box for poster panels."""
    for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        lo, hi = landmarks[:, dim].min() - margin, landmarks[:, dim].max() + margin
        setter(lo, hi)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=15, azim=-60)


def plot_skeleton_poster(ax, landmarks: np.ndarray):
    """Skeleton in the poster palette — bolder lines, halo'd keypoints."""
    for finger, connections in FINGER_CONNECTIONS.items():
        color = POSTER_FINGER_COLORS[finger]
        for i, j in connections:
            ax.plot(
                [landmarks[i, 0], landmarks[j, 0]],
                [landmarks[i, 1], landmarks[j, 1]],
                [landmarks[i, 2], landmarks[j, 2]],
                color=color, linewidth=0.9, solid_capstyle="round", zorder=10,
            )

    for i, j in PALM_CONNECTIONS:
        ax.plot(
            [landmarks[i, 0], landmarks[j, 0]],
            [landmarks[i, 1], landmarks[j, 1]],
            [landmarks[i, 2], landmarks[j, 2]],
            color=POSTER_PALM_COLOR, linewidth=0.7, linestyle=(0, (1, 1.6)), zorder=10,
        )

    # All 21 keypoints marked as circles: fingertips, wrist, inner joints.
    fingertip_idx = [0, 1, 2, 3, 4]
    wrist_idx = [5]
    other_idx = list(range(6, 21))

    ax.scatter(*landmarks[other_idx].T, s=14, c=POSTER_ACCENT_SOFT, zorder=11,
               edgecolors="white", linewidths=0.4)
    ax.scatter(*landmarks[fingertip_idx].T, s=32, c=POSTER_FINGERTIP, zorder=12,
               edgecolors="white", linewidths=0.6)
    ax.scatter(*landmarks[wrist_idx].T, s=44, c=POSTER_ACCENT, zorder=12,
               edgecolors="white", linewidths=0.7)


def plot_mesh_with_skeleton_poster(ax, joint_angles: np.ndarray, mesh_color: str):
    """Translucent pastel mesh with skeleton overlaid, on a clean axis-free 3D panel."""
    vertices, triangles = skin_mesh_from_angles(joint_angles)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color=mesh_color, alpha=0.55,
        edgecolor=mesh_color, linewidth=0.04,
        shade=True, antialiased=True,
    )
    landmarks = joint_angles_to_landmarks(joint_angles)
    plot_skeleton_poster(ax, landmarks)
    style_3d_axis_poster(ax, landmarks)


def plot_mesh_only_poster(ax, joint_angles: np.ndarray, mesh_color: str):
    """Fully opaque pastel mesh — no skeleton overlay, no axes."""
    vertices, triangles = skin_mesh_from_angles(joint_angles)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color=mesh_color, alpha=1.0,
        edgecolor="none", linewidth=0.0,
        shade=True, antialiased=True,
    )
    landmarks = joint_angles_to_landmarks(joint_angles)  # for matching bounds
    style_3d_axis_poster(ax, landmarks)


def plot_skeleton_only_poster(ax, joint_angles: np.ndarray):
    """Skeleton on a clean axis-free 3D panel, framed to match the mesh."""
    landmarks = joint_angles_to_landmarks(joint_angles)
    plot_skeleton_poster(ax, landmarks)
    style_3d_axis_poster(ax, landmarks)


def plot_emg_strip(
    ax,
    emg: np.ndarray,
    start_sec: float,
    end_sec: float,
    frame_times: list[float],
    fs: int = 2000,
    band_half_width_sec: float = 0.04,
):
    """Render 16 stacked rainbow EMG traces with coral highlight bands at frame times.

    Inspired by the gesture/EMG figures from the emg2pose paper.

    emg:  (T, 16) at fs Hz.
    """
    n_channels = emg.shape[1]
    t = np.arange(emg.shape[0]) / fs + start_sec

    # Plot stride keeps line count tractable for big windows.
    stride = max(1, len(t) // 4000)
    t_p = t[::stride]
    emg_p = emg[::stride].astype(np.float32)

    # Normalize each channel to roughly ±0.32 so adjacent traces don't crash into each other.
    scale = np.percentile(np.abs(emg_p), 99, axis=0) + 1e-8
    emg_n = (emg_p / scale) * 0.32

    # Pastel rainbow: HSV wheel pushed toward white for a soft, candy palette.
    cmap = plt.get_cmap("hsv")
    blend = 0.55  # 0 = full saturation, 1 = pure white
    colors = []
    for i in range(n_channels):
        r, g, b, _ = cmap((i + 0.5) / n_channels)
        colors.append((
            r * (1 - blend) + blend,
            g * (1 - blend) + blend,
            b * (1 - blend) + blend,
        ))

    # Channel i sits at y = i + 1 (ch 1 at bottom, ch 16 at top).
    for i in range(n_channels):
        y = (i + 1) + emg_n[:, i]
        ax.plot(t_p, y, color=colors[i], lw=0.55, alpha=0.95, zorder=2)

    # Coral highlight bands at each sampled frame timestamp.
    for ft in frame_times:
        ax.axvspan(
            ft - band_half_width_sec, ft + band_half_width_sec,
            color=POSTER_FINGERTIP, alpha=0.35, lw=0, zorder=1,
        )

    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(0.2, n_channels + 0.8)

    # Channel-number ticks on the left. Only every other channel is labeled
    # so the numbers don't run into each other on a short EMG row.
    shown = list(range(2, n_channels + 1, 2))
    ax.set_yticks(shown)
    ax.set_yticklabels([str(i) for i in shown], fontsize=9)
    ax.tick_params(axis="y", colors="#666666", length=0, pad=2)
    ax.tick_params(axis="x", colors="#666666", labelsize=9, length=3, pad=2)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    ax.set_xlabel("time (s)", fontsize=10, color="#555555", labelpad=3)
    ax.set_ylabel("Channel", fontsize=12, color="#666666", labelpad=6)


def plot_mesh_with_skeleton(ax, joint_angles: np.ndarray, title: str = "",
                            mesh_color: str = "lightblue"):
    """Plot hand mesh with skeleton keypoints overlaid on the same 3D axis."""
    # Draw mesh first (background)
    vertices, triangles = skin_mesh_from_angles(joint_angles)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color=mesh_color, alpha=0.65, edgecolor="gray", linewidth=0.05,
    )
    # Overlay skeleton keypoints on top
    landmarks = joint_angles_to_landmarks(joint_angles)
    plot_skeleton(ax, landmarks, title=title)
    style_3d_axis(ax, landmarks)


# ── Photo + mesh combo (option 1: marketing photo as illustrative GT) ──

def load_marketing_cells(path: str | Path, n_cells: int = 5) -> list[np.ndarray]:
    """Load a strip of hand photos and split it into per-hand RGBA arrays.

    Auto-detects hand bounding boxes by looking for non-background columns,
    then falls back to even slicing if detection fails.
    """
    from PIL import Image
    im = np.array(Image.open(path).convert("RGBA"))
    rgb = im[:, :, :3].astype(np.int32)
    # A column counts as "hand" if any pixel is darker than near-white.
    col_has_hand = (rgb.mean(axis=2).min(axis=0) < 200)

    groups: list[tuple[int, int]] = []
    in_group, start = False, 0
    for i, c in enumerate(col_has_hand):
        if c and not in_group:
            start, in_group = i, True
        elif (not c) and in_group:
            groups.append((start, i))
            in_group = False
    if in_group:
        groups.append((start, len(col_has_hand)))

    if len(groups) != n_cells:
        # Fall back to even split.
        w = im.shape[1]
        step = w // n_cells
        groups = [(i * step, (i + 1) * step) for i in range(n_cells)]

    pad = 4  # tiny breathing room around each detected hand
    cells = []
    for x0, x1 in groups:
        x0 = max(0, x0 - pad)
        x1 = min(im.shape[1], x1 + pad)
        cells.append(im[:, x0:x1])
    return cells


def find_countdown_frames(joint_angles: np.ndarray, fps: int = 30,
                          min_gap_sec: float = 0.4) -> list[int]:
    """Pick five frames whose poses look like 5, 4, 3, 2, 1 extended fingers.

    Strategy: build a continuous "extension score" per frame in [0, 5], smooth
    it, then start from the most-open frame in the whole sequence (target=5)
    and walk forward selecting the frame whose score is closest to 4, then 3,
    then 2, then 1 — enforcing both forward time progression and a minimum
    gap so we don't pick near-duplicate frames.
    """
    T = len(joint_angles)
    dists = np.empty((T, 5), dtype=np.float32)
    for t in range(T):
        lm = joint_angles_to_landmarks(joint_angles[t])
        dists[t] = np.linalg.norm(lm[:5] - lm[5], axis=1)

    lo = np.percentile(dists,  5, axis=0)
    hi = np.percentile(dists, 95, axis=0)
    norm = np.clip((dists - lo) / (hi - lo + 1e-8), 0.0, 1.0)  # (T, 5)
    score = norm.sum(axis=1)                                    # (T,) in [0, 5]

    # Light smoothing so per-frame jitter doesn't pick mid-transition frames.
    w = max(3, fps // 6)
    kernel = np.ones(w) / w
    norm_s = np.stack(
        [np.convolve(norm[:, k], kernel, mode="same") for k in range(5)],
        axis=1,
    )  # (T, 5) smoothed per-finger extension in [0, 1]

    # Per-slot finger patterns matching the marketing photo (left → right):
    #   [thumb, index, middle, ring, pinky], 1 = extended, 0 = curled.
    #   slot "5": all five extended (open hand)
    #   slot "4": thumb tucked, four fingers extended
    #   slot "3": thumb + pinky tucked, three middle fingers extended
    #   slot "2": only index extended (pointing)
    #   slot "1": only thumb extended (thumbs up)
    TARGETS = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],   # 5: all extended (open hand)
        [1.0, 1.0, 1.0, 1.0, 0.0],   # 4: thumb out, pinky tucked
        [1.0, 1.0, 1.0, 0.0, 0.0],   # 3: thumb+index+middle out, ring+pinky tucked
        [0.0, 1.0, 0.0, 0.0, 0.0],   # 2: only index extended (pointing)
        [1.0, 0.0, 0.0, 0.0, 0.0],   # 1: only thumb extended (thumbs up)
    ], dtype=np.float32)

    min_gap = int(min_gap_sec * fps)
    chosen: list[int] = []
    for target_pattern in TARGETS:
        # Squared per-finger error → frame with the closest pattern is best.
        errors = ((norm_s - target_pattern) ** 2).sum(axis=1)
        order = np.argsort(errors)
        picked = None
        for idx in order:
            idx_i = int(idx)
            if all(abs(idx_i - c) > min_gap for c in chosen):
                picked = idx_i
                break
        if picked is None:
            picked = int(order[0])
        chosen.append(picked)
    return chosen


def generate_quad_panel_figure(
    panels: list[dict],
    save_path: str | Path | None = None,
    dpi: int = 200,
    n_frames: int = 5,
    fps: int = 30,
) -> plt.Figure:
    """Compact 2×2 figure. Each panel = title + GT mesh row + REACT mesh row +
    rainbow EMG strip (with bands marking the frame timestamps).

    `panels` is a list of 4 dicts, each containing:
        title:        str
        gt_ja_30hz:   (T, 20) ground-truth joint angles at 30 Hz
        pred_ja_30hz: (T, 20) REACT predictions at 30 Hz
        emg_window:   (T_emg, 16) raw EMG at emg_fs
        emg_fs:       int (Hz)
    """
    assert len(panels) == 4, "quad-panel figure expects exactly 4 panels"
    _setup_poster_style()

    # Portrait-orientation figure: taller than it is wide.
    fig = plt.figure(figsize=(9.0, 12.0), facecolor="white")
    outer_gs = fig.add_gridspec(
        2, 2,
        hspace=0.22, wspace=0.10,
        top=0.95, bottom=0.08, left=0.08, right=0.985,
    )

    for p_idx, panel in enumerate(panels):
        row, col = divmod(p_idx, 2)
        # Three rows inside each panel: GT meshes, prediction meshes, EMG.
        inner = outer_gs[row, col].subgridspec(
            3, n_frames,
            height_ratios=[2.0, 2.0, 2.1],
            hspace=-0.05, wspace=0.0,
        )

        gt_ja = panel["gt_ja_30hz"]
        pred_ja = panel["pred_ja_30hz"]
        emg = panel["emg_window"]
        emg_fs = panel["emg_fs"]

        T = min(len(gt_ja), len(pred_ja))
        # Evenly spaced frame indices through the prediction window.
        frames = [int(T * (i + 0.5) / n_frames) for i in range(n_frames)]
        frame_times = [f / fps for f in frames]

        # Crop EMG to the same time span as the predicted sequence.
        emg_end_sample = int((T / fps) * emg_fs)
        emg_strip = emg[:emg_end_sample]

        # Gesture-name bubble above the panel — pulled close to the GT mesh row.
        panel_bbox = outer_gs[row, col].get_position(fig)
        fig.text(
            (panel_bbox.x0 + panel_bbox.x1) / 2, panel_bbox.y1 - 0.002,
            f"Gesture: {panel['title']}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="round,pad=0.5,rounding_size=0.6",
                facecolor="#0B2564", edgecolor="none",
            ),
        )

        # GT mesh row (blue) and prediction mesh row (pink).
        for c, f in enumerate(frames):
            ax_gt = fig.add_subplot(inner[0, c], projection="3d")
            plot_mesh(ax_gt, gt_ja[f], title="", color=POSTER_GT_MESH)
            style_3d_axis_poster(ax_gt, joint_angles_to_landmarks(gt_ja[f]),
                                 margin=-2.0)
            ax_gt.view_init(elev=12, azim=-86)
            ax_gt.invert_xaxis()

            ax_pr = fig.add_subplot(inner[1, c], projection="3d")
            plot_mesh(ax_pr, pred_ja[f], title="", color=POSTER_PRED_MESH)
            style_3d_axis_poster(ax_pr, joint_angles_to_landmarks(pred_ja[f]),
                                 margin=-2.0)
            ax_pr.view_init(elev=12, azim=-86)
            ax_pr.invert_xaxis()

        # Row labels on the leftmost column of each panel.
        left_x = panel_bbox.x0 - 0.005
        gt_bbox = inner[0, 0].get_position(fig)
        pr_bbox = inner[1, 0].get_position(fig)
        fig.text(left_x, (gt_bbox.y0 + gt_bbox.y1) / 2, "Ground\nTruth",
                 ha="right", va="center", fontsize=9, color=POSTER_TEXT,
                 linespacing=1.1)
        fig.text(left_x, (pr_bbox.y0 + pr_bbox.y1) / 2, "REACT",
                 ha="right", va="center", fontsize=9, color=POSTER_TEXT)

        # EMG strip spanning the full panel width.
        emg_ax = fig.add_subplot(inner[2, :])
        plot_emg_strip(
            emg_ax, emg_strip,
            start_sec=0.0, end_sec=emg_strip.shape[0] / emg_fs,
            frame_times=frame_times, fs=emg_fs,
        )
        # Slim down the EMG labels for compactness.
        emg_ax.set_xlabel("time (s)", fontsize=9, color="#555555", labelpad=2)
        emg_ax.set_ylabel("ch", fontsize=9, color="#666666", labelpad=2)
        emg_ax.tick_params(axis="x", labelsize=8)
        emg_ax.tick_params(axis="y", labelsize=7)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")
    return fig


# Sessions used by --quad-panel. Stop is at 2 kHz samples; titles are display-only.
QUAD_PANEL_SESSIONS = [
    {
        "hdf5":  "src/data/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-4_right.hdf5",
        "title": "Finger Wiggling & Spreading",
        "start": 0, "stop": 60000,
    },
    {
        "hdf5":  "src/data/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-12_right.hdf5",
        "title": "Hook'em, Horns, OK, Scissors",
        "start": 0, "stop": 60000,
    },
    {
        "hdf5":  "src/data/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9_right.hdf5",
        "title": "Individual Finger Pointing & Snap",
        "start": 0, "stop": 60000,
    },
    {
        "hdf5":  "src/data/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-3_right.hdf5",
        "title": "Thumb Swipes & Whole Hand",
        "start": 0, "stop": 60000,
    },
]


def generate_photo_mesh_figure(
    photo_cells: list[np.ndarray],
    pred_joint_angles: np.ndarray,
    frame_indices: list[int],
    save_path: str | Path | None = None,
    dpi: int = 200,
    emg_window: np.ndarray | None = None,
    emg_fs: int = 2000,
    fps: int = 30,
) -> plt.Figure:
    """Top row: marketing photographs. Middle: REACT predicted meshes.
    Optional bottom row: rainbow 16-channel EMG strip with frame markers."""
    assert len(photo_cells) == len(frame_indices)
    n = len(photo_cells)
    has_emg = emg_window is not None and emg_window.size > 0
    fig_h = 7.5 if has_emg else 5.5
    fig = plt.figure(figsize=(4 * n, fig_h), facecolor="white")

    # Per-slot horizontal stretch for the MESH row only — wider columns for
    # the open-hand poses, narrower for the thumbs-up. The photo row stays
    # equal-width so the ground-truth images aren't distorted.
    X_STRETCH = [1.6, 1.6, 1.6, 1.6, 1.0]

    # Two/three independent gridspecs so each row controls its own column widths.
    # Bottom margin reserved for a finger-color legend.
    if has_emg:
        photo_top, photo_bot = 0.95, 0.66
        mesh_top,  mesh_bot  = 0.64, 0.32
        emg_top,   emg_bot   = 0.30, 0.10
    else:
        photo_top, photo_bot = 0.93, 0.52
        mesh_top,  mesh_bot  = 0.50, 0.10
        emg_top,   emg_bot   = None, None

    photo_gs = fig.add_gridspec(
        1, n,
        wspace=0.04,
        top=photo_top, bottom=photo_bot, left=0.02, right=0.98,
    )
    mesh_gs = fig.add_gridspec(
        1, n,
        width_ratios=X_STRETCH,
        wspace=0.04,
        top=mesh_top, bottom=mesh_bot, left=0.02, right=0.98,
    )

    for col, cell in enumerate(photo_cells):
        ax = fig.add_subplot(photo_gs[0, col])
        ax.imshow(cell)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            ax.set_title("Ground Truth", fontsize=16, fontweight="bold",
                         loc="left", pad=10)

    # Per-slot horizontal translation in *data* units (mesh extends ~80 in x).
    # Pure translation of the camera window — the subplot box is untouched,
    # so all meshes render at the same size. Slots 5/4/3 shift leftward,
    # slots 2/1 stay put.
    # Per-slot azimuth (col 0 = "5", col 4 = "1").
    AZIMS = [-86.0, -86.0, -86.0, -86.0, -86.0]
    # Per-slot canvas-space translation, in figure-width fractions. This moves
    # the entire subplot leftward without touching the data or box aspect, so
    # the mesh stays exactly the same size.
    FIG_SHIFTS = [0.030, 0.022, 0.015, 0.007, 0.0]

    for col, fidx in enumerate(frame_indices):
        ax = fig.add_subplot(mesh_gs[0, col], projection="3d")
        ja = pred_joint_angles[fidx]
        plot_mesh(ax, ja, title="", color="lightpink")

        # Overlay the skeleton on top of the (opaque) mesh.
        landmarks = joint_angles_to_landmarks(ja)
        plot_skeleton_poster(ax, landmarks)

        if col == 0:
            ax.set_title("Prediction", fontsize=16, fontweight="bold",
                         pad=2, loc="left")

        style_3d_axis_poster(ax, landmarks, margin=-2.0)

        # Match the data box aspect to the panel's width ratio so the mesh
        # genuinely renders wider in the wider columns instead of just sitting
        # in empty space.
        ax.set_box_aspect((X_STRETCH[col], 1.0, 1.0))

        # Rotate so the fingers extend horizontally (pointing left after the
        # mirror), matching the orientation in the marketing photo.
        ax.view_init(elev=12, azim=AZIMS[col])
        ax.invert_xaxis()

        # Pure canvas-space translation: slide the subplot left on the figure,
        # preserving its width/height so the mesh stays the exact same size.
        shift = FIG_SHIFTS[col]
        if shift != 0:
            pos = ax.get_position()
            ax.set_position((pos.x0 - shift, pos.y0, pos.width, pos.height))

    # Optional rainbow EMG strip below the meshes.
    if has_emg:
        assert emg_window is not None
        emg_gs = fig.add_gridspec(
            1, 1,
            top=emg_top, bottom=emg_bot, left=0.06, right=0.99,
        )
        emg_ax = fig.add_subplot(emg_gs[0, 0])
        start_sec = 0.0
        end_sec = emg_window.shape[0] / emg_fs
        frame_times = [fidx / fps for fidx in frame_indices]
        plot_emg_strip(emg_ax, emg_window, start_sec, end_sec, frame_times,
                       fs=emg_fs)

    # Finger-color legend at the bottom of the figure.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=POSTER_FINGER_COLORS["Thumb"],  lw=3, label="Thumb"),
        Line2D([0], [0], color=POSTER_FINGER_COLORS["Index"],  lw=3, label="Index"),
        Line2D([0], [0], color=POSTER_FINGER_COLORS["Middle"], lw=3, label="Middle"),
        Line2D([0], [0], color=POSTER_FINGER_COLORS["Ring"],   lw=3, label="Ring"),
        Line2D([0], [0], color=POSTER_FINGER_COLORS["Pinky"],  lw=3, label="Pinky"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=POSTER_FINGERTIP, markeredgecolor="white",
               markeredgewidth=0.7, markersize=9, label="Fingertips"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=POSTER_ACCENT, markeredgecolor="white",
               markeredgewidth=0.7, markersize=10, label="Wrist"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=POSTER_ACCENT_SOFT, markeredgecolor="white",
               markeredgewidth=0.6, markersize=7, label="Joints"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=8,
        fontsize=12, frameon=False, handletextpad=0.6, columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.01),
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")
    return fig


def generate_multi_frame_figure(
    gt_joint_angles: np.ndarray,
    pred_joint_angles: np.ndarray | None = None,
    frame_indices: list[int] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 200,
    fps: int = 30,
    emg_window: np.ndarray | None = None,  # accepted for back-compat, unused
    emg_fs: int = 2000,                     # accepted for back-compat, unused
) -> plt.Figure:
    """Two-row × n-column strip of fully opaque hand meshes.

    Top row    = ground truth meshes.
    Bottom row = REACT predictions (falls back to GT if none provided).
    No skeleton overlay, no legend, no EMG strip — just clean meshes on the
    default 3D grid boxes.
    """
    if frame_indices is None:
        n_frames = min(len(gt_joint_angles), 5)
        frames: list[int] = [int(i) for i in np.linspace(
            0, len(gt_joint_angles) - 1, n_frames, dtype=int)]
    else:
        frames = [int(i) for i in frame_indices]

    n = len(frames)
    fig = plt.figure(figsize=(4 * n, 5.5), facecolor="white")
    fig.subplots_adjust(hspace=0.05, wspace=0.05,
                        top=0.92, bottom=0.02, left=0.02, right=0.98)

    for col, fidx in enumerate(frames):
        time_label = f"t = {fidx / fps:.1f}s"

        # Top row — ground truth, opaque pastel blue mesh
        ax_top = fig.add_subplot(2, n, col + 1, projection="3d")
        plot_mesh(ax_top, gt_joint_angles[fidx],
                  title=(f"Ground Truth ({time_label})" if col == 0 else time_label),
                  color="lightblue")
        style_3d_axis_poster(ax_top, joint_angles_to_landmarks(gt_joint_angles[fidx]))

        # Bottom row — prediction (or GT), opaque pastel pink mesh
        ax_bot = fig.add_subplot(2, n, n + col + 1, projection="3d")
        if pred_joint_angles is not None:
            ja = pred_joint_angles[fidx]
            bot_title = f"Prediction ({time_label})" if col == 0 else time_label
            mesh_color = "lightpink"
        else:
            ja = gt_joint_angles[fidx]
            bot_title = f"GT Mesh ({time_label})" if col == 0 else time_label
            mesh_color = "lightblue"
        plot_mesh(ax_bot, ja, title=bot_title, color=mesh_color)
        style_3d_axis_poster(ax_bot, joint_angles_to_landmarks(ja))

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")

    return fig


def run_baseline_inference(session, start=0, stop=10000):
    """Run emg2pose baseline (tracking_vemg2pose) inference — no calibration needed."""
    from emg2pose.utils import generate_hydra_config_from_overrides
    from emg2pose.lightning import Emg2PoseModule

    # Use the pretrained tracking checkpoint
    ckpt_dir = PROJECT_ROOT / "emg2pose_model_checkpoints"
    ckpt = str(ckpt_dir / "tracking_vemg2pose.ckpt")

    config = generate_hydra_config_from_overrides(
        overrides=["experiment=tracking_vemg2pose", f"checkpoint={ckpt}"]
    )
    module = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )
    module.eval()

    window = session[start:stop]
    no_ik = session.no_ik_failure[start:stop]
    batch = {
        "emg": torch.Tensor([window["emg"].T]),
        "joint_angles": torch.Tensor([window["joint_angles"].T]),
        "no_ik_failure": torch.Tensor([no_ik]),
    }

    with torch.no_grad():
        preds, targets, _ = module.forward(batch)

    preds_np = preds[0].T.detach().numpy()
    targets_np = targets[0].T.detach().numpy()
    return preds_np, targets_np


def run_react_inference(session, checkpoint_path, start=0, stop=10000, k=15):
    """Run REACT model inference locally (needs calibration sessions on volume)."""
    import re
    from src.models.hybrid_model import (
        FiLMConditionedModel,
        FiLMConditionedModelConfig,
        load_pretrained_encoder,
        load_pretrained_decoder,
    )
    from src.models.user_encoder import UserEncoderConfig
    from src.models.blocks.gru_pooling import GRUTemporalPoolingConfig
    try:
        from src.models.blocks.cross_attention import CrossAttentionConfig
    except ImportError:
        CrossAttentionConfig = None

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # Architecture checkpoint for encoder/decoder shapes
    ckpt_dir = PROJECT_ROOT / "emg2pose_model_checkpoints"
    raw_path = model_cfg.get("pretrained_checkpoint", "emg2pose_model_checkpoints/tracking_vemg2pose.ckpt")
    encoder_path = str(ckpt_dir / Path(raw_path).name)

    encoder = load_pretrained_encoder(encoder_path, device="cpu")
    decoder = load_pretrained_decoder(encoder_path, device="cpu")

    ue_yaml = model_cfg.get("user_encoder", {})
    pooling = ue_yaml.get("pooling_method", "attention")
    ue_config = UserEncoderConfig(pooling_method=pooling)
    if pooling == "gru":
        gru_yaml = ue_yaml.get("gru_pooling", {})
        if gru_yaml:
            ue_config.gru_pooling = GRUTemporalPoolingConfig(**gru_yaml)

    # Build model config — only include cross_attention/contrastive if available
    config_kwargs = dict(
        feature_dim=model_cfg.get("feature_dim", 64),
        user_embedding_dim=model_cfg.get("user_embedding_dim", 128),
        freeze_encoder=False,
        freeze_decoder=False,
        predict_vel=model_cfg.get("predict_vel", False),
        provide_initial_pos=model_cfg.get("provide_initial_pos", False),
        state_condition=model_cfg.get("state_condition", True),
        user_encoder=ue_config,
    )

    cond_method = model_cfg.get("conditioning_method", "film")
    if CrossAttentionConfig is not None:
        ca_yaml = model_cfg.get("cross_attention", {})
        ca_config = CrossAttentionConfig(
            feature_dim=model_cfg.get("feature_dim", 64),
            **{k_: type(getattr(CrossAttentionConfig, k_, 0))(v) for k_, v in ca_yaml.items() if k_ != "feature_dim"}
        ) if ca_yaml else CrossAttentionConfig(feature_dim=model_cfg.get("feature_dim", 64))
        config_kwargs["conditioning_method"] = cond_method
        config_kwargs["cross_attention"] = ca_config
        config_kwargs["contrastive_dim"] = int(model_cfg.get("contrastive_dim", 0))

    model_config = FiLMConditionedModelConfig(**config_kwargs)
    model = FiLMConditionedModel(model_config, pretrained_encoder=encoder, pretrained_decoder=decoder)

    raw_sd = ckpt["model_state_dict"]
    cleaned_sd = {re.sub(r"\._orig_mod\.", ".", k): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd)
    model.eval()

    # For local single-session inference without calibration pool,
    # use k windows from the *same* session as calibration
    window = session[start:stop]
    emg_tensor = torch.Tensor([window["emg"].T])  # (1, 16, L)

    # Encode calibration windows from other parts of the session
    cal_windows = []
    session_len = len(session)
    cal_window_len = 10000
    for i in range(k):
        cal_start = min(stop + i * cal_window_len, session_len - cal_window_len)
        if cal_start < 0:
            cal_start = 0
        cal_end = cal_start + cal_window_len
        cal_emg = session[cal_start:cal_end]["emg"]  # (L, 16)
        cal_windows.append(torch.from_numpy(cal_emg.T).float())  # (16, L)

    # Encode cal windows with the frozen encoder
    with torch.no_grad():
        cal_features = []
        for cw in cal_windows:
            cf = model.encode(cw.unsqueeze(0))  # (1, C, L')
            cal_features.append(cf.squeeze(0))

        _, C, Lf = cal_features[0].unsqueeze(0).shape
        # Pad to same length
        max_lf = max(cf.shape[-1] for cf in cal_features)
        padded = torch.zeros(1, k, C, max_lf)
        for i, cf in enumerate(cal_features):
            padded[0, i, :, :cf.shape[-1]] = cf

        init_pos = None
        if model_config.provide_initial_pos:
            ja = torch.from_numpy(window["joint_angles"].T).float()  # (20, L)
            left_ctx = getattr(model.encoder, "left_context", 0)
            init_pos = ja[:, left_ctx].unsqueeze(0)  # (1, 20)

        preds = model(
            emg=emg_tensor,
            calibration_features=padded,
            num_calibration_samples=torch.tensor([k]),
            initial_pos=init_pos,
        )

    preds_np = preds[0].T.detach().numpy()  # (T, 20) at rollout_freq (50Hz)
    # Return predictions and the rollout frequency so caller can downsample correctly
    return preds_np, model_config.rollout_freq


def main():
    parser = argparse.ArgumentParser(description="Hand pose comparison visualization")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to emg2pose HDF5 session file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to REACT (.pt) or emg2pose (.ckpt) checkpoint")
    parser.add_argument("--output", type=str, default="paper/final_report/hand_comparison.png")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to visualize (at 30Hz)")
    parser.add_argument("--multi", action="store_true", help="Generate multi-frame strip")
    parser.add_argument("--gt-only", action="store_true", help="Only show ground truth (no model)")
    parser.add_argument("--baseline", action="store_true",
                        help="Use emg2pose baseline (tracking_vemg2pose) for predictions")
    parser.add_argument("--start", type=int, default=0, help="Start sample index (at 2kHz)")
    parser.add_argument("--stop", type=int, default=10000, help="Stop sample index (at 2kHz)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-emg", action="store_true",
                        help="Hide the EMG strip on the multi-frame poster figure.")
    parser.add_argument("--photo-mesh", action="store_true",
                        help="Top row = photographs from --photo; bottom row = REACT mesh predictions.")
    parser.add_argument("--quad-panel", action="store_true",
                        help="Generate a 2×2 figure: 4 gestures, each with GT mesh, REACT mesh, EMG.")
    parser.add_argument("--photo", type=str,
                        default="paper/final_report/marketing_hands.png",
                        help="Path to the marketing-style strip of hand photographs.")
    args = parser.parse_args()

    # Load session data
    print(f"Loading session: {args.hdf5}")
    session = Emg2PoseSessionData(hdf5_path=args.hdf5)

    # Downsample joint angles to 30Hz for visualization
    gt_ja = session["joint_angles"]
    gt_ja_30hz = downsample(gt_ja, native_fs=2000, target_fs=30)
    print(f"Ground truth shape (30Hz): {gt_ja_30hz.shape}")

    pred_ja_30hz = None

    # Grab the raw EMG window for the same span the frames come from.
    # Default span = args.start:args.stop (at 2kHz); falls back to the full
    # session when --gt-only is used and start/stop weren't customized.
    emg_window = session[args.start:args.stop]["emg"]  # (T_emg, 16) @ 2kHz
    emg_fs = 2000

    if args.gt_only:
        # gt_ja_30hz currently spans the full session; align it to the EMG window
        # so frame indices into it correspond to time within the EMG strip.
        gt_ja_30hz = downsample(session[args.start:args.stop]["joint_angles"],
                                native_fs=2000, target_fs=30)
    elif args.baseline:
        print("Running emg2pose baseline inference...")
        preds_np, targets_np = run_baseline_inference(session, args.start, args.stop)
        pred_ja_30hz = downsample(preds_np, native_fs=2000, target_fs=30)
        # Also get aligned GT for the inference window
        gt_ja_30hz = downsample(targets_np, native_fs=2000, target_fs=30)
        print(f"Baseline predictions shape (30Hz): {pred_ja_30hz.shape}")
    elif args.checkpoint:
        if args.checkpoint.endswith(".pt"):
            print(f"Running REACT inference from: {args.checkpoint}")
            preds_np, rollout_freq = run_react_inference(session, args.checkpoint, args.start, args.stop)
            # REACT outputs at rollout_freq (e.g. 50Hz), not 2kHz
            pred_ja_30hz = downsample(preds_np, native_fs=rollout_freq, target_fs=30)
            # GT for the inference window (trim to match prediction timespan)
            window_ja = session[args.start:args.stop]["joint_angles"]
            gt_ja_30hz = downsample(window_ja, native_fs=2000, target_fs=30)
            # Align lengths
            min_len = min(len(gt_ja_30hz), len(pred_ja_30hz))
            gt_ja_30hz = gt_ja_30hz[:min_len]
            pred_ja_30hz = pred_ja_30hz[:min_len]
            print(f"REACT predictions shape (30Hz): {pred_ja_30hz.shape}")
        else:
            # Assume emg2pose lightning checkpoint
            print(f"Running emg2pose inference from: {args.checkpoint}")
            preds_np, targets_np = run_baseline_inference(session, args.start, args.stop)
            pred_ja_30hz = downsample(preds_np, native_fs=2000, target_fs=30)
            gt_ja_30hz = downsample(targets_np, native_fs=2000, target_fs=30)

    # Generate figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.quad_panel:
        if not args.checkpoint:
            raise SystemExit("--quad-panel needs --checkpoint <react.pt>.")

        panels = []
        for sess_cfg in QUAD_PANEL_SESSIONS:
            print(f"\n=== {sess_cfg['title']} ===")
            print(f"  Loading {sess_cfg['hdf5']}")
            sess = Emg2PoseSessionData(hdf5_path=sess_cfg["hdf5"])
            start, stop = sess_cfg["start"], sess_cfg["stop"]

            emg_panel = sess[start:stop]["emg"]            # (T_emg, 16) @ 2 kHz

            print("  Running REACT inference...")
            preds_np, rollout_freq = run_react_inference(
                sess, args.checkpoint, start, stop
            )
            pred_ja_30hz_p = downsample(preds_np, native_fs=rollout_freq, target_fs=30)
            gt_ja_30hz_p = downsample(sess[start:stop]["joint_angles"],
                                      native_fs=2000, target_fs=30)
            T = min(len(gt_ja_30hz_p), len(pred_ja_30hz_p))
            panels.append({
                "title":        sess_cfg["title"],
                "gt_ja_30hz":   gt_ja_30hz_p[:T],
                "pred_ja_30hz": pred_ja_30hz_p[:T],
                "emg_window":   emg_panel,
                "emg_fs":       emg_fs,
            })

        generate_quad_panel_figure(
            panels, save_path=output_path, dpi=args.dpi,
        )
    elif args.photo_mesh:
        if pred_ja_30hz is None:
            raise SystemExit("--photo-mesh needs predictions; pass --checkpoint or --baseline.")
        photo_cells = load_marketing_cells(args.photo, n_cells=5)
        # Detect 5→1 on the *predicted* mesh, so the bottom row visually counts down.
        countdown_frames = find_countdown_frames(pred_ja_30hz)
        print(f"Countdown frames (5→1) at: {countdown_frames}")
        generate_photo_mesh_figure(
            photo_cells,
            pred_ja_30hz,
            frame_indices=countdown_frames,
            save_path=output_path,
            dpi=args.dpi,
            emg_window=None if args.no_emg else emg_window,
            emg_fs=emg_fs,
        )
    elif args.multi:
        generate_multi_frame_figure(
            gt_ja_30hz,
            pred_ja_30hz,
            save_path=output_path,
            dpi=args.dpi,
            emg_window=None if args.no_emg else emg_window,
            emg_fs=emg_fs,
        )
    else:
        generate_comparison_figure(
            gt_ja_30hz,
            pred_ja_30hz,
            frame_idx=args.frame,
            save_path=output_path,
            dpi=args.dpi,
        )

    plt.show()


if __name__ == "__main__":
    main()
