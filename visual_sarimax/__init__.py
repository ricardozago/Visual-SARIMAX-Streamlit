"""Visual SARIMAX Streamlit library."""

from .utils import load_dataset

def main() -> None:
    """Run the Streamlit application."""
    from .app import main as _main

    _main()

__all__ = ["main", "load_dataset"]
