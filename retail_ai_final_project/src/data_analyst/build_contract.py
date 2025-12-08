"""
Dataset Contract Builder
Creates formal data contracts and validates data quality.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_dataset_contract(df: pd.DataFrame, output_path: str = "artifacts/dataset_contract.json") -> Dict[str, Any]:
    """
    Create a formal dataset contract defining schema and constraints.

    Args:
        df: DataFrame to create contract for
        output_path: Path to save the contract JSON

    Returns:
        Dictionary containing the dataset contract
    """
    logger.info("Creating dataset contract")

    contract = {
        "version": "1.0",
        "dataset_name": "retail_ai_dataset",
        "description": "Dataset contract for retail AI project",
        "schema": {},
        "statistics": {},
        "constraints": {},
        "metadata": {
            "total_records": len(df),
            "total_features": len(df.columns),
            "creation_date": pd.Timestamp.now().isoformat()
        }
    }

    # Build schema for each column
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
            "unique_count": int(df[col].nunique()),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percentage": float(df[col].isnull().sum() / len(df) * 100)
        }

        # Add type-specific information
        if df[col].dtype in ['float64', 'int64']:
            col_info.update({
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                "std": float(df[col].std()) if pd.notna(df[col].std()) else None
            })
        elif df[col].dtype == 'object':
            # Get top 10 most frequent values
            top_values = df[col].value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        contract["schema"][col] = col_info

    # Define constraints (customize based on business rules)
    contract["constraints"] = {
        "row_count": {"min": 100, "max": None},
        "missing_threshold": 0.5,  # Maximum 50% missing values per column
        "required_columns": df.columns.tolist()
    }

    # Save contract
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(contract, f, indent=2)

    logger.info(f"Dataset contract saved to {output_path}")

    return contract


def validate_contract(df: pd.DataFrame, contract_path: str = "artifacts/dataset_contract.json") -> bool:
    """
    Validate a DataFrame against a dataset contract.

    Args:
        df: DataFrame to validate
        contract_path: Path to the contract JSON

    Returns:
        Boolean indicating if validation passed
    """
    logger.info("Validating dataset against contract")

    try:
        with open(contract_path, 'r') as f:
            contract = json.load(f)

        validation_errors = []

        # Check required columns
        required_cols = contract["constraints"]["required_columns"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_errors.append(f"Missing required columns: {missing_cols}")

        # Check row count constraints
        min_rows = contract["constraints"]["row_count"]["min"]
        if len(df) < min_rows:
            validation_errors.append(f"Row count {len(df)} below minimum {min_rows}")

        # Check missing value threshold
        missing_threshold = contract["constraints"]["missing_threshold"]
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                validation_errors.append(
                    f"Column {col} exceeds missing threshold: {missing_pct:.2%} > {missing_threshold:.2%}"
                )

        # Check data types
        for col in df.columns:
            if col in contract["schema"]:
                expected_dtype = contract["schema"][col]["dtype"]
                actual_dtype = str(df[col].dtype)
                if expected_dtype != actual_dtype:
                    validation_errors.append(
                        f"Column {col} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                    )

        if validation_errors:
            logger.error("Validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info("Validation passed")
            return True

    except Exception as e:
        logger.error(f"Error validating contract: {str(e)}")
        return False


def main():
    """Main execution function for contract creation."""
    try:
        # Load cleaned data
        df = pd.read_csv("data/interim/clean_data.csv")
        logger.info(f"Loaded cleaned data: {df.shape}")

        # Create contract
        contract = create_dataset_contract(df)

        # Validate contract
        is_valid = validate_contract(df)

        if is_valid:
            logger.info("Dataset contract created and validated successfully")
        else:
            logger.warning("Dataset contract created but validation found issues")

    except Exception as e:
        logger.error(f"Contract creation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
