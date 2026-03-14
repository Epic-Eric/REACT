# REACT-EMG

**FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction**

## Overview

REACT-EMG implements a user-adaptive EMG-to-pose prediction pipeline using Feature-wise Linear Modulation (FiLM) conditioning. The system adapts to individual users through a calibration mechanism that processes k recordings from the same user, enabling personalized hand pose prediction from surface EMG signals.

### Key Features

- **FiLM Conditioning**: User-specific feature modulation via gamma and beta parameters
- **Calibration-Based Adaptation**: Uses k calibration recordings (k in [0, 30]) from the same user for personalization
- **Two Training Modes**: Regression (direct joint angle prediction) and Tracking (velocity-based with LSTM decoder)
- **Two-Phase Training**: Phase 1 with frozen encoder, Phase 2 with layer-wise encoder unfreezing
- **Configurable Architecture**: Hydra-based configuration system with experiment presets
- **Modal Cloud Training**: GPU training on Modal infrastructure (H100) with persistent volumes
- **Full Evaluation Pipeline**: Per-user metrics across 3 generalization splits (User, Stage, User+Stage)

## Architecture

```
EMG (16 channels) --> [Frozen Pretrained TDS Encoder] --> Features (64-dim)
                                                              |
                       Calibration Samples --> [User Encoder] --> User Embedding (128-dim)
                                                              |
                                                        [FiLM Layer] --> gamma, beta
                                                              |
                                                  Features * gamma + beta --> Conditioned Features
                                                              |
                                                [Pretrained LSTM Decoder] --> Joint Angles (20 DOF)
```

### Components

1. **Frozen Encoder**: Pretrained vemg2pose TDS encoder (16 --> 64 channels)
2. **User Encoder Pipeline**:
   - Characteristic CNN (64x3 kernel, extracts temporal patterns)
   - Attention Scorer (1x1 conv, computes sample importance)
   - Transformer Group Encoder (3 layers, 4 heads, aggregates to 128-dim)
3. **FiLM Layer**: Generates gamma and beta from user embedding
4. **Pretrained Decoder**: LSTM decoder from vemg2pose for temporal pose prediction

## Installation

```bash
# Clone repository
git clone <repo-url>
cd REACT

# Install emg2pose submodule (required for pretrained encoder)
pip install -e emg2pose/

# Install react-emg in editable mode
pip install -e .

# With all optional dependencies (dev, viz, modal)
pip install -e ".[all]"
```

### Dependencies

Core: `torch`, `numpy`, `scipy`, `hydra-core`, `omegaconf`, `typer`, `rich`, `tqdm`

Optional groups:
- `dev`: pytest, black, isort, mypy, ruff
- `viz`: matplotlib, seaborn, scikit-learn
- `modal`: modal

## Quick Start

### CLI Usage

```bash
# Show help
react --help

# Train with dummy data (pipeline testing)
react train --dummy --epochs 10

# Train with real data
react train --pretrained /path/to/encoder.ckpt --data-dir /path/to/data

# Show system info
react info
```

### Python API

```python
import torch
from src.models.hybrid_model import FiLMConditionedModel, FiLMConditionedModelConfig

# Create model
config = FiLMConditionedModelConfig(
    feature_dim=64,
    user_embedding_dim=128,
)
model = FiLMConditionedModel(config)
```

## Training

### Local Training

```bash
# Train locally using mini dataset
python scripts/train_local.py
python scripts/train_local.py --config configs/experiment/local.yaml
python scripts/train_local.py --epochs 20
```

### Modal Cloud Training

```bash
# Train on Modal (regression mode, default)
modal run scripts/train_modal.py

# Train in tracking mode
modal run scripts/train_modal.py --config tracking

# With overrides
modal run scripts/train_modal.py --config regression --epochs 100

# List available checkpoints
modal run scripts/train_modal.py --list-ckpts
```

#### Experiment Configs

| Config | File | Mode | Description |
|--------|------|------|-------------|
| `modal_regression` | `configs/experiment/modal_regression.yaml` | Regression | Direct joint angle prediction, no gradient clipping |
| `modal_tracking` | `configs/experiment/modal_tracking.yaml` | Tracking | Velocity prediction with LSTM decoder, gradient clipping |
| `local` | `configs/experiment/local.yaml` | Local | Mini dataset for local development |
| `film_adaptive` | `configs/experiment/film_adaptive.yaml` | Adaptive | FiLM adaptation experiments |

### Two-Phase Training

- **Phase 1** (`num_epochs`): Encoder and decoder are frozen; only FiLM layers, user encoder, and prediction head are trained
- **Phase 2** (`num_epochs_enc_unfreeze`): Encoder layers are gradually unfrozen from output to input, each with layer-wise LR decay

### Calibration Sampling

During training, k calibration recordings are sampled per user (excluding the current session). k is sampled uniformly from `[k_min, k_max]` (default: [0, 30]).

### Loss Function

Combined loss with configurable weights:
- **MAE**: Angular mean absolute error on joint angle predictions
- **Fingertip Distance**: Euclidean distance between predicted and ground truth fingertip positions

## Evaluation

```bash
# Evaluate REACT model on all 3 generalization splits
modal run scripts/eval_modal.py \
    --checkpoint /persistent/react_outputs/run_.../best_model.pt \
    --k 15

# Evaluate with different calibration amounts
modal run scripts/eval_modal.py --checkpoint <path> --k 0
modal run scripts/eval_modal.py --checkpoint <path> --k 5
modal run scripts/eval_modal.py --checkpoint <path> --k 30

# Evaluate baseline vemg2pose (no REACT) for comparison
modal run scripts/eval_modal.py --baseline
modal run scripts/eval_modal.py --baseline --baseline-checkpoint regression_vemg2pose
```

### Evaluation Metrics

Matches the emg2pose paper (Table 5) format:
- **AngleMAE**: Angular mean absolute error (degrees)
- **Per-Finger MAE**: Per-finger breakdown (thumb, index, middle, ring, pinky)
- **Proximal-Distal MAE**: Proximal, mid, distal joint groups
- **Angular Derivatives**: Velocity, acceleration, jerk
- **Landmark Distances**: Fingertip and landmark Euclidean distance (mm)

Results are reported per generalization type:
- **User**: Held-out users, seen stages
- **Stage**: Seen users, held-out stages
- **User, Stage**: Held-out users AND held-out stages

## Project Structure

```
REACT/
├── configs/                        # Hydra configuration files
│   ├── config.yaml                # Main entry point
│   ├── model/
│   │   └── film_conditioned.yaml  # Model architecture config
│   ├── training/
│   │   └── default.yaml           # Training hyperparameters
│   ├── data/
│   │   └── default.yaml           # Data loading config
│   └── experiment/                # Experiment presets
│       ├── modal_regression.yaml  # Modal regression training
│       ├── modal_tracking.yaml    # Modal tracking training
│       ├── local.yaml             # Local development
│       └── film_adaptive.yaml     # FiLM adaptation
├── emg2pose/                       # emg2pose submodule (pretrained encoder/decoder)
├── src/
│   ├── models/                    # Model implementations
│   │   ├── blocks/               # Building blocks
│   │   │   ├── film_layer.py     # FiLM conditioning layer
│   │   │   ├── characteristic_cnn.py
│   │   │   ├── attention_scorer.py
│   │   │   └── transformer_block.py
│   │   ├── user_encoder.py       # User encoder pipeline
│   │   └── hybrid_model.py       # Main FiLM-conditioned model
│   ├── data_loader/              # Data loading utilities
│   │   ├── calibration.py        # Calibration sampling
│   │   └── emg_dataset.py        # User-aware datasets
│   ├── evaluate/                 # Evaluation pipeline
│   │   └── evaluate.py           # Metrics computation (emg2pose-aligned)
│   ├── engine/                   # Training engine
│   │   └── trainer.py
│   ├── utils/                    # Utilities
│   │   ├── data.py               # Dataset classes (PrebuiltCalibratedDataset)
│   │   ├── collate.py            # Custom collate functions
│   │   ├── cache.py              # Dataset caching
│   │   ├── training.py           # Training utilities
│   │   ├── validation.py         # Validation utilities
│   │   ├── signal_proc.py        # EMG signal processing
│   │   ├── visualization.py      # Plotting functions
│   │   └── datasets.py           # Dataset helpers
│   └── cli/                      # Command-line interface
│       ├── main.py
│       └── banner.py
├── scripts/
│   ├── train_modal.py            # Modal cloud training
│   ├── train_local.py            # Local training
│   └── eval_modal.py             # Modal evaluation (REACT + baseline)
├── tests/                        # Unit tests
│   ├── test_blocks.py
│   ├── test_data.py
│   └── test_model.py
├── paper/                        # Paper documents
│   ├── proposal/
│   └── progress_report/
└── pyproject.toml                # Package configuration
```

## Configuration

REACT-EMG uses Hydra for configuration management. Override any parameter from the command line:

```bash
# Override model parameters
react train model.user_embedding_dim=256

# Override training parameters
react train training.num_epochs=200 training.batch_size=64

# Override calibration settings
react train data.calibration.k_min=5 data.calibration.k_max=20
```

## Testing

```bash
pytest                    # Run all tests
pytest tests/test_model.py  # Run model tests only
pytest -v --tb=short      # Verbose with short traceback
```

## Acknowledgments

This project builds upon the [emg2pose](https://github.com/facebookresearch/emg2pose) framework from Meta Research.
