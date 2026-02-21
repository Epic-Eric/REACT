# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
ASCII Banner for REACT-EMG CLI.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


REACT_BANNER = r"""
██████╗ ███████╗ █████╗  ██████╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝╚══██╔══╝
██████╔╝█████╗  ███████║██║        ██║   
██╔══██╗██╔══╝  ██╔══██║██║        ██║   
██║  ██║███████╗██║  ██║╚██████╗   ██║   
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   
"""

REACT_BANNER_SMALL = r"""
 ____  _____    _    ____ _____ 
|  _ \| ____|  / \  / ___|_   _|
| |_) |  _|   / _ \| |     | |  
|  _ <| |___ / ___ \ |___  | |  
|_| \_\_____/_/   \_\____| |_|  
"""

REACT_BANNER_MINIMAL = """
╔══════════════════════════════════════╗
║              R E A C T               ║
║   Real-time EMG Adaptive Control     ║
╚══════════════════════════════════════╝
"""

TAGLINE = "FiLM-Conditioned User-Adaptive EMG-to-Pose Prediction"
VERSION = "1.0.0"


def print_banner(
    console: Console = None,
    style: str = "large",
    show_version: bool = True,
) -> None:
    """Print the REACT ASCII banner.
    
    Args:
        console: Rich console instance.
        style: Banner style ('large', 'small', 'minimal').
        show_version: Whether to show version info.
    """
    if console is None:
        console = Console()
    
    if style == "large":
        banner = REACT_BANNER
    elif style == "small":
        banner = REACT_BANNER_SMALL
    else:
        banner = REACT_BANNER_MINIMAL
    
    # Create styled banner
    banner_text = Text(banner, style="bold cyan")
    
    # Add tagline
    tagline_text = Text(f"\n{TAGLINE}", style="italic white")
    
    if show_version:
        version_text = Text(f"\nVersion {VERSION}", style="dim")
        full_text = banner_text + tagline_text + version_text
    else:
        full_text = banner_text + tagline_text
    
    console.print(full_text, justify="center")
    console.print()


def print_header(
    title: str,
    console: Console = None,
    style: str = "bold magenta",
) -> None:
    """Print a section header.
    
    Args:
        title: Header title.
        console: Rich console instance.
        style: Text style.
    """
    if console is None:
        console = Console()
    
    console.print()
    console.rule(f"[{style}]{title}[/{style}]")
    console.print()


def print_success(message: str, console: Console = None) -> None:
    """Print success message."""
    if console is None:
        console = Console()
    console.print(f"[bold green]✓[/] {message}")


def print_error(message: str, console: Console = None) -> None:
    """Print error message."""
    if console is None:
        console = Console()
    console.print(f"[bold red]✗[/] {message}")


def print_warning(message: str, console: Console = None) -> None:
    """Print warning message."""
    if console is None:
        console = Console()
    console.print(f"[bold yellow]![/] {message}")


def print_info(message: str, console: Console = None) -> None:
    """Print info message."""
    if console is None:
        console = Console()
    console.print(f"[bold blue]ℹ[/] {message}")
