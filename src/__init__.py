# Package metadata
__version__ = '0.1'

# Public API
from .analysis import analyze_data
from .model import train_model, evaluate_model
from .utils import load_data, setup_logging, handle_error, validate_data

__all__ = [
    'analyze_data',
    'train_model',
    'evaluate_model',
    'load_data',
    'setup_logging',
    'handle_error',
    'validate_data'
]

# Initialization code (if any)
setup_logging()
