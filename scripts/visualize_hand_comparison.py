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
        color=color, alpha=0.85, edgecolor="gray", linewidth=0.1,
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


def generate_multi_frame_figure(
    gt_joint_angles: np.ndarray,
    pred_joint_angles: np.ndarray | None = None,
    frame_indices: list[int] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 200,
    fps: int = 30,
) -> plt.Figure:
    """Generate a multi-frame comparison strip.

    Each column shows one timestep.
    Top row = GT mesh + keypoints, bottom row = predicted mesh + keypoints.

    Args:
        fps: Frame rate of the joint angle arrays (default 30Hz after downsampling).
    """
    if frame_indices is None:
        n_frames = min(len(gt_joint_angles), 5)
        frame_indices = np.linspace(0, len(gt_joint_angles) - 1, n_frames, dtype=int).tolist()

    n = len(frame_indices)
    fig = plt.figure(figsize=(4 * n, 5.5), facecolor="white")
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.92, bottom=0.02, left=0.02, right=0.98)

    for col, fidx in enumerate(frame_indices):
        time_sec = fidx / fps
        time_label = f"t = {time_sec:.1f}s"

        gt_ja = gt_joint_angles[fidx]

        # Top row: GT mesh + keypoints
        ax_top = fig.add_subplot(2, n, col + 1, projection="3d")
        top_title = f"Ground Truth ({time_label})" if col == 0 else time_label
        plot_mesh_with_skeleton(ax_top, gt_ja, title=top_title, mesh_color="lightblue")
        if col > 0:
            ax_top.legend().remove()

        # Bottom row: prediction mesh + keypoints
        ax_bot = fig.add_subplot(2, n, n + col + 1, projection="3d")
        if pred_joint_angles is not None:
            pred_ja = pred_joint_angles[fidx]
            color = "lightpink"
            title = f"REACT Prediction ({time_label})" if col == 0 else time_label
        else:
            pred_ja = gt_ja
            color = "lightblue"
            title = f"GT Mesh ({time_label})" if col == 0 else time_label

        plot_mesh_with_skeleton(ax_bot, pred_ja, title=title, mesh_color=color)
        if col > 0:
            ax_bot.legend().remove()

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
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    # Load session data
    print(f"Loading session: {args.hdf5}")
    session = Emg2PoseSessionData(hdf5_path=args.hdf5)

    # Downsample joint angles to 30Hz for visualization
    gt_ja = session["joint_angles"]
    gt_ja_30hz = downsample(gt_ja, native_fs=2000, target_fs=30)
    print(f"Ground truth shape (30Hz): {gt_ja_30hz.shape}")

    pred_ja_30hz = None

    if args.gt_only:
        pass
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

    if args.multi:
        generate_multi_frame_figure(
            gt_ja_30hz,
            pred_ja_30hz,
            save_path=output_path,
            dpi=args.dpi,
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
