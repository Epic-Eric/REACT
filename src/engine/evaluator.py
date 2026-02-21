# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Evaluator for FiLM-Conditioned EMG-to-Pose Model.

Handles inference and evaluation with configurable calibration settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ..models import FiLMConditionedModel
from ..data_loader import CalibrationSampler, CalibrationConfig, UserSessionRegistry


@dataclass
class EvaluatorConfig:
    """Configuration for evaluator.
    
    Attributes:
        calibration_k: Fixed number of calibration samples for inference.
        batch_size: Inference batch size.
        device: Device for inference.
        use_amp: Whether to use automatic mixed precision.
        compute_per_joint_metrics: Compute metrics per joint.
        compute_user_metrics: Compute metrics per user.
    """
    calibration_k: int = 15
    batch_size: int = 32
    device: str = "cuda"
    use_amp: bool = True
    compute_per_joint_metrics: bool = True
    compute_user_metrics: bool = True


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Global metrics
    mae: float = 0.0
    rmse: float = 0.0
    
    # Per-joint metrics
    per_joint_mae: Optional[np.ndarray] = None
    per_joint_rmse: Optional[np.ndarray] = None
    
    # Per-user metrics
    per_user_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Additional info
    num_samples: int = 0
    calibration_k: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'mae': self.mae,
            'rmse': self.rmse,
            'num_samples': self.num_samples,
            'calibration_k': self.calibration_k,
        }
        
        if self.per_joint_mae is not None:
            result['per_joint_mae'] = self.per_joint_mae.tolist()
        
        if self.per_joint_rmse is not None:
            result['per_joint_rmse'] = self.per_joint_rmse.tolist()
        
        if self.per_user_metrics is not None:
            result['per_user_metrics'] = self.per_user_metrics
        
        return result


class Evaluator:
    """Evaluator for inference and metrics computation.
    
    Supports variable calibration settings and per-user analysis.
    """
    
    def __init__(
        self,
        model: FiLMConditionedModel,
        config: EvaluatorConfig,
        registry: Optional[UserSessionRegistry] = None,
    ):
        """Initialize evaluator.
        
        Args:
            model: FiLM-conditioned model.
            config: Evaluator configuration.
            registry: User session registry for calibration.
        """
        self.model = model
        self.config = config
        self.registry = registry
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup calibration sampler if registry provided
        if registry:
            cal_config = CalibrationConfig(
                min_k=config.calibration_k,
                max_k=config.calibration_k,
                sampling_strategy='fixed',
                fixed_k=config.calibration_k,
            )
            self.cal_sampler = CalibrationSampler(cal_config, registry)
        else:
            self.cal_sampler = None
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader,
        k: Optional[int] = None,
    ) -> EvaluationResults:
        """Run full evaluation on test data.
        
        Args:
            test_loader: Test data loader.
            k: Override calibration k value.
        
        Returns:
            EvaluationResults containing metrics.
        """
        k = k or self.config.calibration_k
        
        all_predictions = []
        all_targets = []
        all_user_ids = []
        
        for batch in test_loader:
            batch = self._to_device(batch)
            
            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                pred, target = self._forward_batch(batch)
            
            all_predictions.append(pred.cpu())
            all_targets.append(target.cpu())
            
            if 'user_ids' in batch:
                all_user_ids.extend(batch['user_ids'])
        
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)  # (N, 20, L)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute global metrics
        mae = (predictions - targets).abs().mean().item()
        rmse = ((predictions - targets) ** 2).mean().sqrt().item()
        
        results = EvaluationResults(
            mae=mae,
            rmse=rmse,
            num_samples=predictions.shape[0],
            calibration_k=k,
        )
        
        # Per-joint metrics
        if self.config.compute_per_joint_metrics:
            per_joint_mae = (predictions - targets).abs().mean(dim=(0, 2)).numpy()
            per_joint_rmse = ((predictions - targets) ** 2).mean(dim=(0, 2)).sqrt().numpy()
            results.per_joint_mae = per_joint_mae
            results.per_joint_rmse = per_joint_rmse
        
        # Per-user metrics
        if self.config.compute_user_metrics and all_user_ids:
            results.per_user_metrics = self._compute_per_user_metrics(
                predictions, targets, all_user_ids
            )
        
        return results
    
    @torch.no_grad()
    def predict_single(
        self,
        emg: torch.Tensor,
        calibration_emg: Optional[torch.Tensor] = None,
        session_name: Optional[str] = None,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Make prediction for a single recording.
        
        Args:
            emg: EMG recording (16, L) or (1, 16, L).
            calibration_emg: Optional pre-loaded calibration EMG (K, 16, L_cal).
            session_name: Session name for automatic calibration sampling.
            k: Number of calibration samples.
        
        Returns:
            Pose predictions (20, L') or (1, 20, L').
        """
        k = k or self.config.calibration_k
        
        # Add batch dimension if needed
        if emg.dim() == 2:
            emg = emg.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        emg = emg.to(self.device)
        
        # Get calibration data
        if calibration_emg is not None:
            cal_emg = calibration_emg.unsqueeze(0).to(self.device)
            num_samples = torch.tensor([calibration_emg.shape[0]]).to(self.device)
        elif session_name and self.cal_sampler:
            cal_emg, num_samples, lengths = self.cal_sampler.get_calibration_batch(
                session_name, k=k, device=self.device
            )
            cal_emg = cal_emg.unsqueeze(0)
            num_samples = num_samples.unsqueeze(0)
        else:
            # No calibration - use zero embedding
            cal_emg = torch.zeros(1, 1, 16, 100).to(self.device)
            num_samples = torch.tensor([0]).to(self.device)
        
        # Encode calibration
        B, K, C, L = cal_emg.shape
        cal_flat = cal_emg.view(B * K, C, L)
        cal_features_flat = self.model.encode(cal_flat)
        _, C_out, L_out = cal_features_flat.shape
        cal_features = cal_features_flat.view(B, K, C_out, L_out)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            predictions = self.model(emg, cal_features, num_samples)
        
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions.cpu()
    
    @torch.no_grad()
    def evaluate_calibration_effect(
        self,
        test_loader,
        k_values: List[int] = [0, 5, 10, 15, 20, 25, 30],
    ) -> Dict[int, EvaluationResults]:
        """Evaluate model performance across different k values.
        
        Useful for analyzing how calibration amount affects accuracy.
        
        Args:
            test_loader: Test data loader.
            k_values: List of k values to evaluate.
        
        Returns:
            Dictionary mapping k to EvaluationResults.
        """
        results = {}
        
        for k in k_values:
            print(f"Evaluating with k={k}...")
            results[k] = self.evaluate(test_loader, k=k)
        
        return results
    
    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Forward pass for a batch."""
        # Get main EMG and targets
        emg = batch['emg']
        joint_angles = batch['joint_angles']
        
        # Get calibration data
        cal_emg = batch['calibration_emg']
        num_cal_samples = batch['num_calibration_samples']
        cal_lengths = batch.get('calibration_lengths')
        
        # Encode calibration
        B, K, C, L = cal_emg.shape
        cal_flat = cal_emg.view(B * K, C, L)
        cal_features_flat = self.model.encode(cal_flat)
        _, C_out, L_out = cal_features_flat.shape
        cal_features = cal_features_flat.view(B, K, C_out, L_out)
        
        # Adjust lengths
        if cal_lengths is not None and L_out > 0:
            ds_factor = L / L_out
            cal_lengths = (cal_lengths / ds_factor).long().clamp(min=1)
        
        # Forward
        predictions = self.model(emg, cal_features, num_cal_samples, cal_lengths)
        
        # Align targets
        left_ctx = self.model.left_context
        right_ctx = self.model.right_context
        if left_ctx > 0 or right_ctx > 0:
            start = left_ctx
            end = -right_ctx if right_ctx > 0 else None
            joint_angles = joint_angles[..., start:end]
        
        min_len = min(predictions.shape[-1], joint_angles.shape[-1])
        predictions = predictions[..., :min_len]
        joint_angles = joint_angles[..., :min_len]
        
        return predictions, joint_angles
    
    def _compute_per_user_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        user_ids: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per user."""
        unique_users = list(set(user_ids))
        user_metrics = {}
        
        for user in unique_users:
            mask = [u == user for u in user_ids]
            user_pred = predictions[mask]
            user_target = targets[mask]
            
            mae = (user_pred - user_target).abs().mean().item()
            rmse = ((user_pred - user_target) ** 2).mean().sqrt().item()
            
            user_metrics[user] = {
                'mae': mae,
                'rmse': rmse,
                'num_samples': user_pred.shape[0],
            }
        
        return user_metrics
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result


class InferenceEngine:
    """Simplified inference interface for production use."""
    
    def __init__(
        self,
        checkpoint_path: str,
        pretrained_encoder_path: str,
        registry: Optional[UserSessionRegistry] = None,
        device: str = "cuda",
    ):
        """Load model from checkpoint for inference.
        
        Args:
            checkpoint_path: Path to REACT-EMG checkpoint.
            pretrained_encoder_path: Path to pretrained encoder checkpoint.
            registry: User session registry.
            device: Inference device.
        """
        from ..models import load_pretrained_encoder, FiLMConditionedModelConfig
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load pretrained encoder
        encoder = load_pretrained_encoder(pretrained_encoder_path, device=str(self.device))
        
        # Create model
        config = FiLMConditionedModelConfig()
        self.model = FiLMConditionedModel(config, pretrained_encoder=encoder)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup evaluator
        eval_config = EvaluatorConfig(device=str(self.device))
        self.evaluator = Evaluator(self.model, eval_config, registry)
    
    def predict(
        self,
        emg: np.ndarray,
        calibration_emg: Optional[np.ndarray] = None,
        k: int = 15,
    ) -> np.ndarray:
        """Make pose predictions from EMG.
        
        Args:
            emg: EMG signal (16, L) or (L, 16).
            calibration_emg: Optional calibration EMG (K, 16, L_cal).
            k: Number of calibration samples to use.
        
        Returns:
            Joint angle predictions (20, L').
        """
        # Handle input format
        if emg.shape[0] != 16:
            emg = emg.T
        
        emg_tensor = torch.from_numpy(emg).float()
        
        if calibration_emg is not None:
            if calibration_emg.shape[1] != 16:
                calibration_emg = calibration_emg.transpose(0, 2, 1)
            cal_tensor = torch.from_numpy(calibration_emg).float()
        else:
            cal_tensor = None
        
        predictions = self.evaluator.predict_single(emg_tensor, cal_tensor, k=k)
        
        return predictions.numpy()
