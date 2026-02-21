# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Trainer for FiLM-Conditioned EMG-to-Pose Model.

Handles the full training loop including:
- Encoding calibration recordings
- Computing user embeddings
- FiLM conditioning
- Loss computation and backpropagation
- Logging and checkpointing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from ..models import FiLMConditionedModel, FiLMConditionedModelConfig


@dataclass
class TrainerConfig:
    """Configuration for trainer.
    
    Attributes:
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        max_epochs: Maximum training epochs.
        gradient_clip_val: Maximum gradient norm (None for no clipping).
        warmup_epochs: Number of warmup epochs for scheduler.
        scheduler_type: Type of learning rate scheduler.
        log_every_n_steps: Logging frequency.
        val_every_n_epochs: Validation frequency.
        save_every_n_epochs: Checkpoint save frequency.
        early_stopping_patience: Epochs to wait for improvement.
        checkpoint_dir: Directory for saving checkpoints.
        resume_from: Path to checkpoint to resume from.
        mixed_precision: Whether to use mixed precision training.
        compile_model: Whether to use torch.compile (PyTorch 2.0+).
    """
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    gradient_clip_val: Optional[float] = 1.0
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "none"
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 20
    checkpoint_dir: Path = Path("checkpoints")
    resume_from: Optional[Path] = None
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class TrainingState:
    """Mutable training state."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    epochs_without_improvement: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


class Trainer:
    """Trainer for FiLM-conditioned EMG-to-pose model.
    
    Handles the complete training loop with gradient computation
    only for trainable components (user encoder, FiLM, prediction head).
    """
    
    def __init__(
        self,
        model: FiLMConditionedModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Callable] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: FiLM-conditioned model to train.
            config: Trainer configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            logger: Optional logging function.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or print
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile if requested (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Setup optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        
        # Initialize state
        self.state = TrainingState()
        
        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume if requested
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.max_epochs,
                steps_per_epoch=len(self.train_loader),
            )
        return None
    
    def train(self) -> Dict[str, Any]:
        """Run full training loop.
        
        Returns:
            Dictionary with training results and metrics.
        """
        self.logger(f"Starting training on {self.device}")
        self.logger(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        
        for epoch in range(self.state.epoch, self.config.max_epochs):
            self.state.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            self.state.train_losses.append(train_metrics['loss'])
            
            # Validation
            if self.val_loader and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate()
                self.state.val_losses.append(val_metrics['loss'])
                
                # Check for improvement
                if val_metrics['loss'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics['loss']
                    self.state.epochs_without_improvement = 0
                    self.save_checkpoint('best.pt')
                else:
                    self.state.epochs_without_improvement += 1
                
                self.logger(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Best: {self.state.best_val_loss:.4f}"
                )
                
                # Early stopping
                if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                self.logger(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}"
                )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
            
            # Update scheduler
            if self.scheduler and self.config.scheduler_type == "cosine":
                self.scheduler.step()
        
        training_time = time.time() - start_time
        
        # Save final checkpoint
        self.save_checkpoint('final.pt')
        
        return {
            'best_val_loss': self.state.best_val_loss,
            'final_train_loss': self.state.train_losses[-1] if self.state.train_losses else None,
            'epochs_trained': self.state.epoch + 1,
            'training_time_seconds': training_time,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.
        
        Returns:
            Dictionary with epoch metrics.
        """
        self.model.train()
        
        # Ensure encoder stays in eval mode if frozen
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            if self.model.config.freeze_encoder:
                self.model.encoder.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                predictions, targets = self._forward_batch(batch)
                losses = self._compute_loss(predictions, targets, batch.get('no_ik_failure'))
                loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val,
                    )
                
                self.optimizer.step()
            
            # Update OneCycle scheduler per step
            if self.scheduler and self.config.scheduler_type == "onecycle":
                self.scheduler.step()
            
            total_loss += loss.item()
            total_mae += losses['mae'].item()
            num_batches += 1
            self.state.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                self.logger(f"  Step {batch_idx+1}/{len(self.train_loader)} - Loss: {avg_loss:.4f}")
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = self._to_device(batch)
            
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                predictions, targets = self._forward_batch(batch)
                losses = self._compute_loss(predictions, targets, batch.get('no_ik_failure'))
            
            total_loss += losses['total_loss'].item()
            total_mae += losses['mae'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
            'mae': total_mae / num_batches if num_batches > 0 else float('inf'),
        }
    
    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Forward pass for a batch.
        
        Handles encoding of calibration data and main forward pass.
        """
        # Get main EMG and targets
        emg = batch['emg']  # (B, 16, L)
        joint_angles = batch['joint_angles']  # (B, 20, L)
        
        # Get calibration data
        cal_emg = batch['calibration_emg']  # (B, K_max, 16, L_cal)
        num_cal_samples = batch['num_calibration_samples']  # (B,)
        cal_lengths = batch.get('calibration_lengths')  # (B, K_max)
        
        # Encode calibration EMG with frozen encoder
        B, K_max, C_in, L_cal = cal_emg.shape
        
        with torch.no_grad():
            cal_emg_flat = cal_emg.view(B * K_max, C_in, L_cal)
            cal_features_flat = self.model.encode(cal_emg_flat)
            _, C_out, L_out = cal_features_flat.shape
            cal_features = cal_features_flat.view(B, K_max, C_out, L_out)
        
        # Adjust lengths for downsampling
        if cal_lengths is not None:
            ds_factor = L_cal / L_out if L_out > 0 else 1
            cal_lengths = (cal_lengths / ds_factor).long().clamp(min=1)
        
        # Forward pass through model
        predictions = self.model(
            emg=emg,
            calibration_features=cal_features,
            num_calibration_samples=num_cal_samples,
            calibration_lengths=cal_lengths,
        )
        
        # Align targets with predictions (account for encoder context)
        left_ctx = self.model.left_context
        right_ctx = self.model.right_context
        if left_ctx > 0 or right_ctx > 0:
            start = left_ctx
            end = -right_ctx if right_ctx > 0 else None
            joint_angles = joint_angles[..., start:end]
        
        # Ensure same temporal length
        min_len = min(predictions.shape[-1], joint_angles.shape[-1])
        predictions = predictions[..., :min_len]
        joint_angles = joint_angles[..., :min_len]
        
        return predictions, joint_angles
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        # Ensure matching lengths
        min_len = min(predictions.shape[-1], targets.shape[-1])
        predictions = predictions[..., :min_len]
        targets = targets[..., :min_len]
        
        if mask is not None:
            mask = mask[..., :min_len]
            diff = (predictions - targets).abs()
            diff = diff * mask.unsqueeze(1)
            mae = diff.sum() / (mask.sum() * predictions.shape[1] + 1e-8)
            
            diff_sq = (predictions - targets) ** 2
            diff_sq = diff_sq * mask.unsqueeze(1)
            mse = diff_sq.sum() / (mask.sum() * predictions.shape[1] + 1e-8)
        else:
            mae = F.l1_loss(predictions, targets)
            mse = F.mse_loss(predictions, targets)
        
        return {
            'mae': mae,
            'mse': mse,
            'total_loss': mae,  # Primary loss
        }
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state': {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_val_loss': self.state.best_val_loss,
                'epochs_without_improvement': self.state.epochs_without_improvement,
                'train_losses': self.state.train_losses,
                'val_losses': self.state.val_losses,
            },
            'config': self.config.__dict__,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        state_dict = checkpoint['state']
        self.state.epoch = state_dict['epoch']
        self.state.global_step = state_dict['global_step']
        self.state.best_val_loss = state_dict['best_val_loss']
        self.state.epochs_without_improvement = state_dict['epochs_without_improvement']
        self.state.train_losses = state_dict.get('train_losses', [])
        self.state.val_losses = state_dict.get('val_losses', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger(f"Resumed from checkpoint: {path} (epoch {self.state.epoch+1})")


class DummyTrainer:
    """Simplified trainer for testing with dummy data.
    
    Validates the full pipeline without complex data loading.
    """
    
    def __init__(
        self,
        model: FiLMConditionedModel,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
    
    def run_dummy_training(
        self,
        num_steps: int = 10,
        batch_size: int = 4,
        seq_length: int = 2000,
        k_calibration: int = 5,
    ) -> Dict[str, Any]:
        """Run dummy training to verify pipeline.
        
        Args:
            num_steps: Number of training steps.
            batch_size: Batch size.
            seq_length: EMG sequence length.
            k_calibration: Number of calibration samples.
        
        Returns:
            Results dictionary.
        """
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        
        losses = []
        
        for step in range(num_steps):
            # Generate dummy data
            emg = torch.randn(batch_size, 16, seq_length).to(self.device)
            targets = torch.randn(batch_size, 20, seq_length).to(self.device)
            
            cal_emg = torch.randn(batch_size, k_calibration, 16, 1000).to(self.device)
            num_samples = torch.full((batch_size,), k_calibration).to(self.device)
            cal_lengths = torch.full((batch_size, k_calibration), 1000).to(self.device)
            
            # Encode calibration with frozen encoder
            B, K, C, L = cal_emg.shape
            with torch.no_grad():
                cal_flat = cal_emg.view(B * K, C, L)
                cal_features_flat = self.model.encode(cal_flat)
                _, C_out, L_out = cal_features_flat.shape
                cal_features = cal_features_flat.view(B, K, C_out, L_out)
            
            # Forward
            pred = self.model(emg, cal_features, num_samples)
            
            # Align shapes
            min_len = min(pred.shape[-1], targets.shape[-1])
            pred = pred[..., :min_len]
            targets = targets[..., :min_len]
            
            # Loss
            loss = F.l1_loss(pred, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"Step {step+1}/{num_steps} - Loss: {loss.item():.4f}")
        
        return {
            'losses': losses,
            'final_loss': losses[-1],
        }
