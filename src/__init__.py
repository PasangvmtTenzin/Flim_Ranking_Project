# Package metadata
__version__ = '0.1'

# Public API
from .analysis import load_data, calculate_weak_impact, calculate_strong_impact, analyze_quality_by_country
from .analysis import plot_and_save_weak_impact, plot_and_save_strong_impact, plot_and_save_quality_by_country
from .analysis import perform_analysis

# List of public symbols
__all__ = [
    'load_data',
    'calculate_weak_impact',
    'calculate_strong_impact',
    'analyze_quality_by_country',
    'plot_and_save_weak_impact',
    'plot_and_save_strong_impact',
    'plot_and_save_quality_by_country',
    'perform_analysis'
]

# Initialization code (if any)
from .utils import setup_logging, handle_error, validate_data

# Ensure logging setup when package is imported
setup_logging()
