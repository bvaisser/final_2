"""
Data Ingestion and Cleaning Module
Loads raw data and performs comprehensive cleaning operations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_dataframe

logger = get_logger(__name__)


def load_data(file_path: str = "data/raw/Coffe_sales.csv") -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        file_path: Path to the raw CSV file

    Returns:
        DataFrame containing the raw data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by handling missing values, duplicates, and data types.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning process")
    df_clean = df.copy()

    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows")

    # Handle missing values
    missing_summary = df_clean.isnull().sum()
    if missing_summary.sum() > 0:
        logger.info(f"Missing values found:\n{missing_summary[missing_summary > 0]}")

        # Strategy: Fill numeric columns with median, categorical with mode
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    logger.info(f"Filled missing values in {col} with median")
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    logger.info(f"Filled missing values in {col} with mode")

    # Basic data type corrections (customize based on your dataset)
    # Example: Convert date columns if present
    date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
            logger.info(f"Converted {col} to datetime")
        except Exception as e:
            logger.warning(f"Could not convert {col} to datetime: {str(e)}")

    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean


def save_clean_data(df: pd.DataFrame, output_path: str = "data/interim/clean_data.csv") -> None:
    """
    Save cleaned data to CSV file.

    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data
    """
    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise


def main():
    """Main execution function for data ingestion and cleaning."""
    try:
        # Load data
        df = load_data()

        # Clean data
        df_clean = clean_data(df)

        # Validate cleaned data
        if validate_dataframe(df_clean):
            logger.info("Data validation passed")

        # Save cleaned data
        save_clean_data(df_clean)

        logger.info("Data ingestion and cleaning completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
