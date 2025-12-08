"""
Validation Utilities
Provides data validation and error checking functions.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from .logging_utils import get_logger

logger = get_logger(__name__)


def validate_file_exists(file_path: str, file_type: str = "file") -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file
        file_type: Type of file for error message (default: "file")

    Returns:
        True if file exists

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        error_msg = f"{file_type.capitalize()} not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"{file_type.capitalize()} exists: {file_path}")
    return True


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: Optional[List[str]] = None,
    check_nulls: bool = True,
    null_threshold: float = 0.5
) -> bool:
    """
    Validate a DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names
        check_nulls: Whether to check for excessive null values
        null_threshold: Maximum proportion of nulls allowed per column

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating DataFrame")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for excessive nulls
    if check_nulls:
        null_percentages = df.isnull().sum() / len(df)
        excessive_nulls = null_percentages[null_percentages > null_threshold]

        if not excessive_nulls.empty:
            logger.warning(f"Columns with excessive nulls (>{null_threshold*100}%):")
            for col, pct in excessive_nulls.items():
                logger.warning(f"  - {col}: {pct*100:.2f}%")

    logger.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_column_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str]
) -> bool:
    """
    Validate that columns have expected data types.

    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types

    Returns:
        True if validation passes

    Raises:
        ValueError: If column types don't match
    """
    logger.info("Validating column types")

    type_mismatches = []

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            type_mismatches.append(f"Column '{col}' not found in DataFrame")
            continue

        actual_type = str(df[col].dtype)

        # Normalize type names for comparison
        if expected_type in ['int', 'integer', 'int64']:
            expected_type = 'int64'
        elif expected_type in ['float', 'float64']:
            expected_type = 'float64'
        elif expected_type in ['str', 'string', 'object']:
            expected_type = 'object'

        if not actual_type.startswith(expected_type):
            type_mismatches.append(
                f"Column '{col}': expected {expected_type}, got {actual_type}"
            )

    if type_mismatches:
        error_msg = "Column type validation failed:\n" + "\n".join(type_mismatches)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Column type validation passed")
    return True


def validate_value_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> bool:
    """
    Validate that column values fall within expected range.

    Args:
        df: DataFrame to validate
        column: Column name to check
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        True if validation passes

    Raises:
        ValueError: If values are out of range
    """
    logger.info(f"Validating value range for column: {column}")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    col_data = df[column].dropna()

    if min_value is not None:
        below_min = col_data < min_value
        if below_min.any():
            count = below_min.sum()
            raise ValueError(
                f"Column '{column}': {count} values below minimum {min_value}"
            )

    if max_value is not None:
        above_max = col_data > max_value
        if above_max.any():
            count = above_max.sum()
            raise ValueError(
                f"Column '{column}': {count} values above maximum {max_value}"
            )

    logger.info(f"Value range validation passed for column: {column}")
    return True


def validate_no_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> bool:
    """
    Validate that DataFrame has no duplicate rows.

    Args:
        df: DataFrame to validate
        subset: Optional list of columns to check for duplicates

    Returns:
        True if no duplicates found

    Raises:
        ValueError: If duplicates are found
    """
    logger.info("Checking for duplicate rows")

    duplicates = df.duplicated(subset=subset)
    dup_count = duplicates.sum()

    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate rows")

    logger.info("No duplicate rows found")
    return True


def validate_categories(
    df: pd.DataFrame,
    column: str,
    allowed_values: List[Any]
) -> bool:
    """
    Validate that categorical column only contains allowed values.

    Args:
        df: DataFrame to validate
        column: Column name to check
        allowed_values: List of allowed values

    Returns:
        True if validation passes

    Raises:
        ValueError: If invalid values are found
    """
    logger.info(f"Validating categories for column: {column}")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    unique_values = df[column].dropna().unique()
    invalid_values = set(unique_values) - set(allowed_values)

    if invalid_values:
        raise ValueError(
            f"Column '{column}' contains invalid values: {invalid_values}"
        )

    logger.info(f"Category validation passed for column: {column}")
    return True
