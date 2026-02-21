```
██████╗ ███████╗ █████╗  ██████╗████████╗      ███████╗███╗   ███╗ ██████╗ 
██╔══██╗██╔════╝██╔══██╗██╔════╝╚══██╔══╝      ██╔════╝████╗ ████║██╔════╝ 
██████╔╝█████╗  ███████║██║        ██║         █████╗  ██╔████╔██║██║  ███╗
██╔══██╗██╔══╝  ██╔══██║██║        ██║         ██╔══╝  ██║╚██╔╝██║██║   ██║
██║  ██║███████╗██║  ██║╚██████╗   ██║         ███████╗██║ ╚═╝ ██║╚██████╔╝
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝         ╚══════╝╚═╝     ╚═╝ ╚═════╝ 
```

**FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction**

## Overview

REACT-EMG implements a user-adaptive EMG-to-pose prediction pipeline using Feature-wise Linear Modulation (FiLM) conditioning. The system adapts to individual users through a calibration mechanism that processes k recordings from the same user.

### Key Features

- **FiLM Conditioning**: User-specific feature modulation via gamma and beta parameters
- **Calibration-Based Adaptation**: Uses k∈[0,30] recordings from the same user for personalization
- **Configurable Architecture**: Hydra-based configuration system
- **Modal Cloud Training**: Ready for distributed training on Modal infrastructure
- **Rich CLI**: Beautiful terminal interface with progress tracking

## Architecture

```
EMG (16 channels) → [Frozen Pretrained Encoder] → Features (64-dim)
                                                        ↓
                         Calibration Samples → [User Encoder] → User Embedding (128-dim)
                                                        ↓
                                                  [FiLM Layer] → γ, β
                                                        ↓
                                            Features × γ + β → Conditioned Features
                                                        ↓
                                              [Prediction Head] → Joint Angles (20 DOF)
```

### Components

1. **Frozen Encoder**: Pretrained vemg2pose TDS encoder (16 → 64 channels)
2. **User Encoder Pipeline**:
   - Characteristic CNN (64×3 kernel, extracts temporal patterns)
   - Attention Scorer (1×1 conv, computes sample importance)
   - Transformer Group Encoder (3 layers, 4 heads, aggregates to 128-dim)
3. **FiLM Layer**: Generates γ and β from user embedding
4. **Prediction Head**: MLP outputting 20 DOF joint angles

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/react-emg.git
cd react-emg

# Install emg2pose submodule (required for pretrained encoder)
pip install -e emg2pose/

# Install react-emg in editable mode
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```
pip install -e ".[all]"
```

## Quick Start

### CLI Usage

```bash
# Show help
react --help

# Train with dummy data (pipeline testing)
react train --dummy --epochs 10

# Train with real data
react train --pretrained /path/to/encoder.ckpt --data-dir /path/to/data

# Evaluate model
react evaluate checkpoint.pt --data-dir /path/to/test_data

# Show system info
react info
```

### Python API

```python
import torch
from src.models.hybrid_model import FiLMConditionedModel, FiLMConditionedModelConfig
from src.utils.data import create_dummy_batch, create_dataloaders

# Create model
config = FiLMConditionedModelConfig(
    feature_dim=64,
    user_embedding_dim=128,
    num_joints=20,
)
model = FiLMConditionedModel(config)

# Create dummy batch (for quick testing without data)
batch = create_dummy_batch(batch_size=4, calibration_k=10)

# Forward pass
output = model(
    encoded_features=batch["emg"],  # Would be encoder output
    calibration_features=batch["calibration_emg"],
)
print(output["predictions"].shape)  # (4, 20, 10000)

# Load real data (requires emg2pose_dataset_mini)
# train_loader, val_loader, test_loader = create_dataloaders()
```

### Modal Cloud Training

```bash
# Train on Modal with GPU (uses emg2pose_dataset_mini)
modal run scripts/modal_train.py --epochs 100

# List available checkpoints
modal run scripts/modal_train.py --list-ckpts
```

## Project Structure

```
REACT/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main entry point
│   ├── model/                 # Model configurations
│   ├── training/              # Training configurations
│   ├── data/                  # Data configurations
│   └── experiment/            # Experiment presets
├── emg2pose/                   # emg2pose submodule (pretrained encoder)
│   ├── emg2pose/              # Core emg2pose package
│   │   ├── networks.py        # TDS encoder architecture
│   │   ├── data.py            # Dataset classes
│   │   └── ...
│   └── setup.py               # Install with: pip install -e emg2pose/
├── src/
│   ├── models/                # Model implementations
│   │   ├── blocks/           # Building blocks
│   │   │   ├── film_layer.py
│   │   │   ├── characteristic_cnn.py
│   │   │   ├── attention_scorer.py
│   │   │   └── transformer_block.py
│   │   ├── user_encoder.py   # User encoder pipeline
│   │   └── hybrid_model.py   # Main FiLM-conditioned model
│   ├── data_loader/          # Data loading utilities
│   │   ├── calibration.py    # Calibration sampling
│   │   └── emg_dataset.py    # User-aware datasets
│   ├── engine/               # Training/evaluation engines
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── utils/                # Utilities
│   │   ├── signal_proc.py    # EMG signal processing
│   │   ├── visualization.py  # Plotting functions
│   │   └── dummy_data.py     # Synthetic data generation
│   └── cli/                  # Command-line interface
│       ├── main.py
│       └── banner.py
├── scripts/
│   └── modal_train.py        # Modal cloud training
├── tests/                    # Unit tests
├── pyproject.toml           # Package configuration
└── README.md
```

## Configuration

REACT-EMG uses Hydra for configuration management. Override any parameter from the command line:

```bash
# Override model parameters
react train model.user_embedding_dim=256 model.film_layer.num_layers=3

# Override training parameters
react train training.num_epochs=200 training.batch_size=64

# Override calibration settings
react train data.calibration.k_min=5 data.calibration.k_max=20
```

## Training

### Calibration Sampling

During training, for each sample, k calibration recordings are sampled from the same user (excluding the current session). The sampling strategy is configurable:

- **uniform**: Uniformly sample k from [k_min, k_max]
- **truncated_normal**: Sample k from truncated normal distribution
- **fixed**: Always use a fixed k value
- **weighted**: Weight sampling by session similarity

### Loss Function

The model is trained with MSE loss on joint angle predictions:

```
L = MSE(predicted_angles, target_angles)
```

The encoder remains frozen during training; only the user encoder, FiLM layer, and prediction head are optimized.

## Evaluation

Evaluate across different calibration amounts to measure adaptation effectiveness:

```bash
react evaluate checkpoint.pt --data-dir /path/to/test --calibration-k 0
react evaluate checkpoint.pt --data-dir /path/to/test --calibration-k 10
react evaluate checkpoint.pt --data-dir /path/to/test --calibration-k 30
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@software{react_emg,
  title = {REACT-EMG: FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction},
  year = {2024},
  url = {https://github.com/yourusername/react-emg}
}
```

## Acknowledgments

This project builds upon the [emg2pose](https://github.com/facebookresearch/emg2pose) framework from Meta Research.
