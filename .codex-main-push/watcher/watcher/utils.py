"""
Utility functions for Watcher.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger("watcher")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root.
    """
    # This file is at watcher/watcher/utils.py
    # Project root is two levels up
    return Path(__file__).parent.parent.parent


def resolve_path(path: str, relative_to: Path = None) -> Path:
    """
    Resolve a path, making it absolute if relative.
    
    Args:
        path: Path string (may be relative).
        relative_to: Base directory for relative paths. Defaults to project root.
        
    Returns:
        Absolute Path object.
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    
    if relative_to is None:
        relative_to = get_project_root()
    
    return relative_to / path_obj


def print_source_info(info) -> None:
    """
    Print formatted source information.
    
    Args:
        info: SourceInfo object.
    """
    print("\n" + "=" * 50)
    print("SOURCE INFORMATION")
    print("=" * 50)
    print(f"Source Type:      {info.source_type.value}")
    print(f"Source String:    {info.source_string}")
    print(f"Opened:           {'YES' if info.is_opened else 'NO'}")
    if info.is_opened:
        print(f"Resolution:       {info.width} x {info.height}")
        print(f"FPS:              {info.fps:.2f}")
        if info.frame_count > 0:
            print(f"Frame Count:      {info.frame_count}")
        else:
            print(f"Frame Count:      N/A (live stream)")
    else:
        print(f"Error:            {info.error_message}")
    print("=" * 50 + "\n")


def format_fps(fps: float) -> str:
    """Format FPS for display."""
    if fps > 0:
        return f"{fps:.2f}"
    return "N/A"


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object for the directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj