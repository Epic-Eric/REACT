#!/usr/bin/env python3
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Local training script for REACT-EMG.

Loads configuration from configs/experiment/local.yaml.
Trains on emg2pose_dataset_mini with logging and visualization.

Usage:
    python scripts/train_local.py
    python scripts/train_local.py --config configs/experiment/local.yaml
    python scripts/train_local.py --epochs 20  # Override config value
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Local imports
from src.utils.training import (
    load_config,
    TrainingLogger,
    plot_training_curves,
    plot_predictions,
    get_device,
    save_checkpoint,
    save_history,
    compute_loss_with_length_match,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train REACT-EMG model locally")
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(ROOT / "configs/experiment/local.yaml"),
        help="Path to config file"
    )
    # Allow overriding common params via CLI
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(Path(args.config))
    
    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["optimizer"]["lr"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    
    # Extract config values (with explicit type conversion for YAML quirks)
    num_epochs = int(cfg["training"]["num_epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["optimizer"]["lr"])
    weight_decay = float(cfg["training"]["optimizer"]["weight_decay"])
    grad_clip = float(cfg["training"]["gradient_clip_norm"])
    num_workers = int(cfg["training"]["num_workers"])
    
    k_min = int(cfg["data"]["calibration"]["k_min"])
    k_max = int(cfg["data"]["calibration"]["k_max"])
    
    feature_dim = int(cfg["model"]["feature_dim"])
    user_embedding_dim = int(cfg["model"]["user_embedding_dim"])
    freeze_encoder = bool(cfg["model"]["freeze_encoder"])
    checkpoint_rel_path = str(cfg["model"]["pretrained_checkpoint"])
    
    seed = cfg.get("seed", 42)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(cfg["output"]["dir"])
    output_dir = output_base / f"run_{timestamp}"
    
    # Setup logging
    logger = TrainingLogger(output_dir)
    
    # Device selection
    device = get_device(cfg["device"])
    
    # Log header and config
    logger.log("=" * 60)
    logger.log("REACT-EMG Local Training")
    logger.log("=" * 60)
    logger.log(f"Config file: {args.config}")
    logger.log(f"PyTorch version: {torch.__version__}")
    logger.log(f"Device: {device}")
    logger.log(f"Output dir: {output_dir}")
    logger.log("")
    logger.log("Training parameters:")
    logger.log(f"  Epochs: {num_epochs}")
    logger.log(f"  Batch size: {batch_size}")
    logger.log(f"  Learning rate: {lr}")
    logger.log(f"  Weight decay: {weight_decay}")
    logger.log(f"  Gradient clip: {grad_clip}")
    logger.log("")
    logger.log("Calibration parameters:")
    logger.log(f"  K min: {k_min}")
    logger.log(f"  K max: {k_max}")
    logger.log("")
    logger.log("Model parameters:")
    logger.log(f"  Feature dim: {feature_dim}")
    logger.log(f"  User embedding dim: {user_embedding_dim}")
    logger.log(f"  Freeze encoder: {freeze_encoder}")
    logger.log("=" * 60)
    
    # Import model and data modules
    from src.utils.data import create_dataloaders
    from src.models.hybrid_model import (
        FiLMConditionedModel, 
        FiLMConditionedModelConfig,
        load_pretrained_encoder,
    )
    
    # Check checkpoint
    checkpoint_path = ROOT / checkpoint_rel_path
    if not checkpoint_path.exists():
        logger.log(f"ERROR: Pretrained checkpoint not found: {checkpoint_path}")
        logger.log("Please download vemg2pose checkpoint from emg2pose releases.")
        return 1
    logger.log(f"\nPretrained checkpoint: {checkpoint_path}")
    
    # Data directory
    data_dir = ROOT / cfg["data"]["data_dir"]
    logger.log(f"Dataset location: {data_dir}")
    if not data_dir.exists():
        logger.log("ERROR: Dataset not found!")
        return 1
    
    # Create data loaders
    logger.log("\nLoading data...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            calibration_k=k_max,
            min_calibration_k=k_min,
            window_length=int(cfg["data"]["window_length"]),
            stride=int(cfg["data"]["stride"]),
        )
        logger.log(f"Train batches: {len(train_loader)}")
        logger.log(f"Val batches: {len(val_loader)}")
        logger.log(f"Test batches: {len(test_loader)}")
    except Exception as e:
        logger.log(f"ERROR loading data: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1
    
    # Load pretrained encoder
    logger.log("\nLoading pretrained encoder from vemg2pose...")
    try:
        pretrained_encoder = load_pretrained_encoder(
            checkpoint_path=str(checkpoint_path),
            device=str(device),
        )
        logger.log("Encoder loaded successfully.")
    except Exception as e:
        logger.log(f"ERROR loading encoder: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1
    
    # Create model
    logger.log("\nCreating model...")
    model_config = FiLMConditionedModelConfig(
        feature_dim=feature_dim,
        user_embedding_dim=user_embedding_dim,
        freeze_encoder=freeze_encoder,
    )
    model = FiLMConditionedModel(model_config, pretrained_encoder=pretrained_encoder).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {num_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=tuple(float(b) for b in cfg["training"]["optimizer"]["betas"]),
        eps=float(cfg["training"]["optimizer"]["eps"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=float(cfg["training"]["scheduler"]["min_lr"]),
    )
    criterion = nn.MSELoss()
    
    # Training loop
    logger.log("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            emg = batch["emg"].to(device)
            targets = batch["joint_angles"].to(device)
            calibration_emg = batch["calibration_emg"].to(device)
            num_cal_samples = batch["calibration_k"].to(device)
            
            optimizer.zero_grad()
            
            try:
                predictions = model.forward_with_raw_calibration(
                    emg=emg,
                    calibration_emg=calibration_emg,
                    num_calibration_samples=num_cal_samples,
                )
                
                loss = compute_loss_with_length_match(predictions, targets, criterion)
                loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_losses.append(loss.item())
                
            except Exception as e:
                logger.log(f"Error in batch {batch_idx}: {e}")
                import traceback
                logger.log(traceback.format_exc())
                continue
        
        avg_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                emg = batch["emg"].to(device)
                targets = batch["joint_angles"].to(device)
                calibration_emg = batch["calibration_emg"].to(device)
                num_cal_samples = batch["calibration_k"].to(device)
                
                try:
                    predictions = model.forward_with_raw_calibration(
                        emg=emg,
                        calibration_emg=calibration_emg,
                        num_calibration_samples=num_cal_samples,
                    )
                    
                    loss = compute_loss_with_length_match(predictions, targets, criterion)
                    val_epoch_losses.append(loss.item())
                except:
                    continue
        
        avg_val_loss = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Logging
        logger.log(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            save_checkpoint(
                output_dir / "best_model.pt",
                epoch, model, optimizer, best_val_loss, cfg,
            )
            logger.log(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
        
        # Plot training curves periodically
        plot_every = cfg["output"].get("plot_every", 5)
        if (epoch + 1) % plot_every == 0 or epoch == num_epochs - 1:
            plot_training_curves(
                train_losses, val_losses,
                output_dir / "training_curves.png"
            )
    
    total_time = time.time() - start_time
    
    # Final logging
    logger.log("\n" + "=" * 60)
    logger.log("Training Complete!")
    logger.log("=" * 60)
    logger.log(f"Total time: {total_time / 60:.1f} minutes")
    logger.log(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    logger.log(f"Final train loss: {train_losses[-1]:.4f}")
    logger.log(f"Final val loss: {val_losses[-1]:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        output_dir / "final_model.pt",
        num_epochs, model, optimizer, val_losses[-1], cfg,
        extra={"train_losses": train_losses, "val_losses": val_losses},
    )
    logger.log(f"Final model saved to: {output_dir / 'final_model.pt'}")
    
    # Generate final visualizations
    logger.log("\nGenerating visualizations...")
    
    plot_training_curves(
        train_losses, val_losses,
        output_dir / "training_curves_final.png"
    )
    
    # Plot example predictions
    if cfg["output"].get("save_predictions", True):
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            emg = batch["emg"].to(device)
            targets = batch["joint_angles"].to(device)
            calibration_emg = batch["calibration_emg"].to(device)
            num_cal_samples = batch["calibration_k"].to(device)
            
            predictions = model.forward_with_raw_calibration(
                emg=emg,
                calibration_emg=calibration_emg,
                num_calibration_samples=num_cal_samples,
            )
            
            pred_len = predictions.shape[-1]
            target_len = targets.shape[-1]
            min_len = min(pred_len, target_len)
            
            plot_predictions(
                emg, targets[..., :min_len], predictions[..., :min_len],
                output_dir / "predictions_example.png"
            )
    
    # Save training history
    save_history(
        output_dir / "history.json",
        train_losses, val_losses, best_val_loss, best_epoch, total_time, cfg,
    )
    
    # =================================================================
    # Evaluation (emg2pose-style metrics on train, val, test)
    # =================================================================
    logger.log("\n" + "=" * 60)
    logger.log("Running emg2pose-style evaluation...")
    logger.log("=" * 60)
    
    # Load best model for eval
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    logger.log(f"Loaded best model from epoch {best_ckpt['epoch'] + 1}")
    
    from emg2pose.constants import JOINTS, NUM_JOINTS, LANDMARKS, NO_MOVEMENT_LANDMARKS
    from emg2pose.kinematics import forward_kinematics, load_default_hand_model, TorchHandModel
    
    hand_model = TorchHandModel(load_default_hand_model())
    
    def evaluate_loader(loader, split_name):
        """Evaluate on a loader using emg2pose AngleMAE metric."""
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in loader:
                emg = batch["emg"].to(device)
                targets = batch["joint_angles"].to(device)
                calibration_emg = batch["calibration_emg"].to(device)
                num_cal_samples = batch["calibration_k"].to(device)
                
                predictions = model.forward_with_raw_calibration(
                    emg=emg,
                    calibration_emg=calibration_emg,
                    num_calibration_samples=num_cal_samples,
                )
                
                # Align lengths
                min_len = min(predictions.shape[-1], targets.shape[-1])
                pred = predictions[..., :min_len]
                tgt = targets[..., :min_len]
                
                # IK failure mask
                if "no_ik_failure" in batch:
                    mask = batch["no_ik_failure"].to(device)[..., :min_len]
                else:
                    mask = torch.ones(tgt.shape[0], min_len, dtype=torch.bool, device=device)
                
                all_preds.append(pred.cpu())
                all_targets.append(tgt.cpu())
                all_masks.append(mask.cpu())
        
        preds = torch.cat(all_preds, dim=0)     # (N, 20, L)
        targets = torch.cat(all_targets, dim=0)  # (N, 20, L)
        masks = torch.cat(all_masks, dim=0)       # (N, L)
        
        # AngleMAE (same as emg2pose AngleMAE metric)
        mask_exp = masks.unsqueeze(1).expand_as(preds)  # (N, 20, L)
        overall_mae = torch.nn.L1Loss()(preds[mask_exp], targets[mask_exp]).item()
        
        # MSE
        overall_mse = torch.nn.MSELoss()(preds[mask_exp], targets[mask_exp]).item()
        
        # Per-joint MAE
        per_joint_mae = {}
        for j in JOINTS:
            j_mask = masks  # (N, L)
            j_pred = preds[:, j.index]  # (N, L)
            j_tgt = targets[:, j.index]
            if j_mask.any():
                per_joint_mae[j.name] = torch.nn.L1Loss()(j_pred[j_mask], j_tgt[j_mask]).item()
        
        # Per-finger MAE
        fingers = {"thumb": [], "index": [], "middle": [], "ring": [], "pinky": []}
        for j in JOINTS:
            for finger in fingers:
                if finger in j.groups:
                    fingers[finger].append(j.index)
        per_finger_mae = {}
        for finger, idxs in fingers.items():
            f_mask = masks.unsqueeze(1).expand(-1, len(idxs), -1)
            f_pred = preds[:, idxs]
            f_tgt = targets[:, idxs]
            if f_mask.any():
                per_finger_mae[finger] = torch.nn.L1Loss()(f_pred[f_mask], f_tgt[f_mask]).item()
        
        # Landmark / fingertip distances (same as emg2pose LandmarkDistances metric)
        # Downsample by 40x to avoid OOM in forward_kinematics (same as emg2pose)
        ds = 40
        sl = slice(None, None, ds)
        eval_device = torch.device("cpu")  # FK can be memory-heavy
        
        if hand_model.device != eval_device:
            hand_model.to(eval_device)
        
        pred_pos = forward_kinematics(preds[:, :, sl].to(eval_device), hand_model)
        tgt_pos = forward_kinematics(targets[:, :, sl].to(eval_device), hand_model)
        mask_sl = masks[:, sl].to(eval_device)
        
        # Fingertip distance
        ft_idxs = [lm.index for lm in LANDMARKS if "fingertip" in lm.groups]
        ft_mask = mask_sl[..., None].expand(-1, -1, len(ft_idxs))
        fingertip_dist = torch.linalg.norm(
            pred_pos[:, :, ft_idxs] - tgt_pos[:, :, ft_idxs], dim=-1
        )[ft_mask].mean().item()
        
        # Landmark distance (all moving landmarks)
        lm_idxs = [lm.index for lm in LANDMARKS if lm.name not in NO_MOVEMENT_LANDMARKS]
        lm_mask = mask_sl[..., None].expand(-1, -1, len(lm_idxs))
        landmark_dist = torch.linalg.norm(
            pred_pos[:, :, lm_idxs] - tgt_pos[:, :, lm_idxs], dim=-1
        )[lm_mask].mean().item()
        
        # Print results (FK outputs in mm since hand model is in mm)
        logger.log(f"\n--- {split_name} ---")
        logger.log(f"  Samples: {preds.shape[0]}")
        logger.log(f"  AngleMAE:           {overall_mae:.6f} rad  ({np.degrees(overall_mae):.4f} deg)")
        logger.log(f"  MSE:                {overall_mse:.6f}")
        logger.log(f"  RMSE:               {np.sqrt(overall_mse):.6f} rad  ({np.degrees(np.sqrt(overall_mse)):.4f} deg)")
        logger.log(f"  Fingertip distance: {fingertip_dist:.2f} mm")
        logger.log(f"  Landmark distance:  {landmark_dist:.2f} mm")
        
        logger.log(f"  Per-finger MAE (deg):")
        for finger, mae in per_finger_mae.items():
            logger.log(f"    {finger:10s}: {np.degrees(mae):.4f}")
        
        logger.log(f"  Per-joint MAE (deg):")
        for name, mae in per_joint_mae.items():
            logger.log(f"    {name:25s}: {np.degrees(mae):.4f}")
        
        return {"mae": overall_mae, "mae_deg": np.degrees(overall_mae), "mse": overall_mse,
                "fingertip_mm": fingertip_dist, "landmark_mm": landmark_dist}
    
    train_metrics = evaluate_loader(train_loader, "TRAIN")
    val_metrics = evaluate_loader(val_loader, "VAL")
    test_metrics = evaluate_loader(test_loader, "TEST")
    
    logger.log(f"\n{'='*60}")
    logger.log(f"Summary:")
    logger.log(f"  {'':10s} {'MAE (deg)':>10s} {'Fingertip':>12s} {'Landmark':>12s}")
    logger.log(f"  {'Train':10s} {train_metrics['mae_deg']:10.4f} {train_metrics['fingertip_mm']:10.2f} mm {train_metrics['landmark_mm']:10.2f} mm")
    logger.log(f"  {'Val':10s} {val_metrics['mae_deg']:10.4f} {val_metrics['fingertip_mm']:10.2f} mm {val_metrics['landmark_mm']:10.2f} mm")
    logger.log(f"  {'Test':10s} {test_metrics['mae_deg']:10.4f} {test_metrics['fingertip_mm']:10.2f} mm {test_metrics['landmark_mm']:10.2f} mm")
    logger.log(f"{'='*60}")
    
    logger.log(f"\nAll outputs saved to: {output_dir}")
    logger.log("Files:")
    logger.log("  - training.log")
    logger.log("  - training_curves_final.png")
    logger.log("  - predictions_example.png")
    logger.log("  - best_model.pt")
    logger.log("  - final_model.pt")
    logger.log("  - history.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
