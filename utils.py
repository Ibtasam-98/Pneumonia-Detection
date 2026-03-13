"""Utility functions for the Chest X-Ray ML Predictor"""
import time
from typing import Optional
from datetime import datetime


def print_header(text: str, width: int = 80):
    """Print a formatted header"""
    print(f"\n{'=' * width}")
    print(f"{text.center(width)}")
    print(f"{'=' * width}")


def print_success(text: str):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message"""
    print(f"❌ {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_info(text: str):
    """Print info message"""
    print(f"ℹ️  {text}")


class TimingManager:
    """Context manager for timing code blocks"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.description}...")
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"⏱️  Completed: {self.description} in {elapsed:.2f} seconds")


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str):
    """Ensure directory exists"""
    import os
    os.makedirs(directory, exist_ok=True)


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a table string"""
    from tabulate import tabulate

    table_data = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table_data.append([key, f"{value:.4f}"])
        else:
            table_data.append([key, str(value)])

    return tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")