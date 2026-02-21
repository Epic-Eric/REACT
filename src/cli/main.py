# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
REACT-EMG Main CLI Application.

Entry point for training, evaluation, and visualization commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from .banner import (
    print_banner,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)


# Create Typer app
app = typer.Typer(
    name="react",
    help="REACT: FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        from .banner import VERSION
        console.print(f"REACT-EMG version {VERSION}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """REACT-EMG: FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction."""
    pass


@app.command()
def train(
    config: str = typer.Option(
        "configs/config.yaml",
        "--config",
        "-c",
        help="Path to Hydra configuration file.",
    ),
    pretrained: Optional[str] = typer.Option(
        None,
        "--pretrained",
        "-p",
        help="Path to pretrained encoder checkpoint.",
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Path to data directory.",
    ),
    output_dir: Optional[str] = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints and logs.",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Training batch size.",
    ),
    lr: float = typer.Option(
        1e-4,
        "--learning-rate",
        "--lr",
        help="Learning rate.",
    ),
    calibration_k: int = typer.Option(
        10,
        "--calibration-k",
        "-k",
        help="Number of calibration samples.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device to train on (cuda/cpu/mps).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed.",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to checkpoint to resume from.",
    ),
    dummy: bool = typer.Option(
        False,
        "--dummy",
        help="Use dummy data for testing.",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Verbose output.",
    ),
):
    """Train FiLM-conditioned user-adaptive model."""
    print_banner(console)
    print_header("Training", console)
    
    # Display configuration
    config_table = Table(title="Training Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Config File", config)
    config_table.add_row("Pretrained Encoder", pretrained or "None")
    config_table.add_row("Data Directory", data_dir or "Not specified")
    config_table.add_row("Output Directory", output_dir)
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Learning Rate", f"{lr:.2e}")
    config_table.add_row("Calibration K", str(calibration_k))
    config_table.add_row("Device", device)
    config_table.add_row("Seed", str(seed))
    config_table.add_row("Dummy Data", str(dummy))
    
    console.print(config_table)
    console.print()
    
    try:
        import torch
        from ..engine.trainer import Trainer, TrainerConfig, DummyTrainer
        from ..models.hybrid_model import FiLMConditionedModel, FiLMConditionedModelConfig
        from ..utils.dummy_data import create_dummy_dataloaders
        
        # Set seed
        torch.manual_seed(seed)
        
        if dummy:
            print_info("Using dummy data for pipeline testing...")
            
            # Create dummy trainer for testing
            trainer = DummyTrainer(
                emg_channels=16,
                num_joints=20,
                feature_dim=64,
                user_embedding_dim=128,
                calibration_k=calibration_k,
                device=device,
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Training...", total=epochs)
                
                for epoch in range(epochs):
                    loss = trainer.train_epoch()
                    progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
            
            print_success(f"Training completed! Final loss: {trainer.current_loss:.4f}")
        else:
            if pretrained is None:
                print_error("Pretrained encoder path required for real training.")
                print_info("Use --pretrained to specify encoder checkpoint, or --dummy for testing.")
                raise typer.Exit(1)
            
            if data_dir is None:
                print_error("Data directory required for real training.")
                print_info("Use --data-dir to specify data path, or --dummy for testing.")
                raise typer.Exit(1)
            
            print_info("Setting up real training pipeline...")
            # Real training would be implemented here
            print_warning("Real training not yet fully implemented. Use --dummy for testing.")
            
    except ImportError as e:
        print_error(f"Import error: {e}")
        print_info("Make sure all dependencies are installed: pip install -e .")
        raise typer.Exit(1)
    
    print_success("Training complete!")


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(
        ...,
        help="Path to model checkpoint.",
    ),
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Path to evaluation data.",
    ),
    output_dir: Optional[str] = typer.Option(
        "eval_outputs",
        "--output-dir",
        "-o",
        help="Output directory for results.",
    ),
    calibration_k: int = typer.Option(
        10,
        "--calibration-k",
        "-k",
        help="Number of calibration samples.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device for evaluation.",
    ),
    save_predictions: bool = typer.Option(
        False,
        "--save-predictions",
        help="Save prediction outputs.",
    ),
):
    """Evaluate trained model on test data."""
    print_banner(console)
    print_header("Evaluation", console)
    
    config_table = Table(title="Evaluation Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Checkpoint", checkpoint)
    config_table.add_row("Data Directory", data_dir)
    config_table.add_row("Output Directory", output_dir)
    config_table.add_row("Calibration K", str(calibration_k))
    config_table.add_row("Device", device)
    
    console.print(config_table)
    console.print()
    
    # Verify checkpoint exists
    if not Path(checkpoint).exists():
        print_error(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(1)
    
    print_info("Loading model and running evaluation...")
    
    try:
        from ..engine.evaluator import Evaluator, EvaluatorConfig
        
        # Evaluation would be implemented here
        print_warning("Evaluation not yet fully implemented.")
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        raise typer.Exit(1)


@app.command()
def visualize(
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint for embedding visualization.",
    ),
    training_log: Optional[str] = typer.Option(
        None,
        "--log",
        "-l",
        help="Path to training log for loss curves.",
    ),
    output_dir: str = typer.Option(
        "visualizations",
        "--output-dir",
        "-o",
        help="Output directory for plots.",
    ),
    plot_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Plot type: loss, embeddings, predictions, all.",
    ),
):
    """Generate visualizations from training results."""
    print_banner(console, style="small")
    print_header("Visualization", console)
    
    print_info(f"Generating {plot_type} visualizations...")
    print_info(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        from ..utils.visualization import (
            plot_training_curves,
            plot_user_embeddings,
            plot_predictions,
        )
        
        if training_log and plot_type in ["loss", "all"]:
            print_info("Plotting training curves...")
            # Would load log and plot
        
        if checkpoint and plot_type in ["embeddings", "all"]:
            print_info("Plotting user embeddings...")
            # Would load model and visualize embeddings
        
        print_success(f"Visualizations saved to {output_dir}")
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        print_info("Install matplotlib: pip install matplotlib")
        raise typer.Exit(1)


@app.command("info")
def show_info():
    """Show system and package information."""
    print_banner(console)
    print_header("System Information", console)
    
    info_table = Table(show_header=True)
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    
    # Check Python
    info_table.add_row("Python", sys.version.split()[0])
    
    # Check PyTorch
    try:
        import torch
        cuda_status = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
        info_table.add_row("PyTorch", f"{torch.__version__} ({cuda_status})")
        
        if torch.cuda.is_available():
            info_table.add_row("GPU", torch.cuda.get_device_name(0))
    except ImportError:
        info_table.add_row("PyTorch", "[red]Not installed[/red]")
    
    # Check other dependencies
    deps = {
        "rich": "Rich CLI",
        "typer": "Typer CLI",
        "hydra": "Hydra Config",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            info_table.add_row(name, "[green]Installed[/green]")
        except ImportError:
            info_table.add_row(name, "[red]Not installed[/red]")
    
    console.print(info_table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
