import pandas as pd
import logging
import os

def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise IOError(f"An error occurred while reading the file {file_path}: {e}")

def validate_data(data, required_columns):
    """Validates that the required columns are present in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing: {', '.join(missing_columns)}")

def setup_logging(log_file='app.log'):
    """Sets up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Logging setup complete.")

def handle_error(e, msg="An error occurred"):
    """Logs an error message and raises the exception."""
    logging.error(f"{msg}: {e}")
    raise e

