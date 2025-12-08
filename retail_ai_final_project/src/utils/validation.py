"""
Validation Utilities for Retail AI Pipeline
Provides contract validation, data quality checks, and business rules validation.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class DatasetContractValidator:
    """
    Validates datasets against contract specifications.
    Ensures data quality and schema compliance between crew handoffs.
    """

    def __init__(self, contract_path: str):
        """
        Initialize validator with contract file.

        Args:
            contract_path: Path to dataset_contract.json
        """
        self.contract_path = Path(contract_path)
        self.contract = self._load_contract()

    def _load_contract(self) -> Dict:
        """Load and parse the dataset contract."""
        if not self.contract_path.exists():
            raise ValidationError(
                f"Dataset contract not found: {self.contract_path}"
            )

        try:
            with open(self.contract_path, 'r') as f:
                contract = json.load(f)
            return contract
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in contract file: {e}")

    def validate_file_exists(self, file_path: str) -> bool:
        """
        Check if a required file exists.

        Args:
            file_path: Path to file to check

        Returns:
            True if file exists

        Raises:
            ValidationError if file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"Required file not found: {file_path}")

        if path.stat().st_size == 0:
            raise ValidationError(f"File is empty: {file_path}")

        return True

    def validate_dataset_schema(self, dataset_path: str) -> bool:
        """
        Validate dataset schema matches contract specifications.

        Args:
            dataset_path: Path to CSV dataset

        Returns:
            True if schema is valid

        Raises:
            ValidationError if schema doesn't match
        """
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            raise ValidationError(f"Failed to load dataset: {e}")

        # Get contract schema
        if 'schema' not in self.contract:
            raise ValidationError("Contract missing 'schema' field")

        expected_schema = self.contract['schema']

        # Check all required columns exist
        required_columns = set(expected_schema.keys())
        actual_columns = set(df.columns)

        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}"
            )

        # Validate data types for each column
        for col_name, col_spec in expected_schema.items():
            if col_name not in df.columns:
                continue

            expected_dtype = col_spec.get('dtype')
            actual_dtype = str(df[col_name].dtype)

            # Map pandas dtypes to contract dtypes
            if expected_dtype:
                if not self._dtypes_match(actual_dtype, expected_dtype):
                    raise ValidationError(
                        f"Column '{col_name}' type mismatch: "
                        f"expected {expected_dtype}, got {actual_dtype}"
                    )

        return True

    def _dtypes_match(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        dtype_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'bool': ['bool'],
            'datetime': ['datetime64[ns]']
        }

        for dtype_family, dtypes in dtype_mapping.items():
            if expected == dtype_family and actual in dtypes:
                return True
            if expected in dtypes and actual in dtypes:
                return True

        return actual == expected

    def validate_data_quality(self, dataset_path: str) -> bool:
        """
        Validate data quality requirements from contract.

        Args:
            dataset_path: Path to CSV dataset

        Returns:
            True if quality checks pass

        Raises:
            ValidationError if quality checks fail
        """
        df = pd.read_csv(dataset_path)

        # Get quality requirements
        quality_req = self.contract.get('quality_requirements', {})

        # Check minimum row count
        min_rows = quality_req.get('min_rows', 0)
        if len(df) < min_rows:
            raise ValidationError(
                f"Dataset has {len(df)} rows, minimum required: {min_rows}"
            )

        # Check missing value thresholds
        max_missing_pct = quality_req.get('max_missing_percentage', 100)
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > max_missing_pct:
                raise ValidationError(
                    f"Column '{col}' has {missing_pct:.2f}% missing values, "
                    f"maximum allowed: {max_missing_pct}%"
                )

        # Check for duplicate rows
        allow_duplicates = quality_req.get('allow_duplicates', True)
        if not allow_duplicates:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                raise ValidationError(
                    f"Dataset contains {duplicate_count} duplicate rows"
                )

        return True

    def validate_business_rules(self, dataset_path: str) -> bool:
        """
        Validate business logic rules from contract.

        Args:
            dataset_path: Path to CSV dataset

        Returns:
            True if business rules pass

        Raises:
            ValidationError if business rules fail
        """
        df = pd.read_csv(dataset_path)

        # Get business rules
        business_rules = self.contract.get('business_rules', [])

        for rule in business_rules:
            rule_type = rule.get('type')
            column = rule.get('column')

            if column not in df.columns:
                raise ValidationError(
                    f"Business rule references non-existent column: {column}"
                )

            # Value range validation
            if rule_type == 'range':
                min_val = rule.get('min')
                max_val = rule.get('max')

                if min_val is not None:
                    violations = df[df[column] < min_val]
                    if len(violations) > 0:
                        raise ValidationError(
                            f"Column '{column}' has {len(violations)} values "
                            f"below minimum {min_val}"
                        )

                if max_val is not None:
                    violations = df[df[column] > max_val]
                    if len(violations) > 0:
                        raise ValidationError(
                            f"Column '{column}' has {len(violations)} values "
                            f"above maximum {max_val}"
                        )

            # Allowed values validation
            elif rule_type == 'allowed_values':
                allowed = set(rule.get('values', []))
                actual = set(df[column].unique())
                invalid = actual - allowed

                if invalid:
                    raise ValidationError(
                        f"Column '{column}' contains invalid values: {invalid}"
                    )

            # Not null validation
            elif rule_type == 'not_null':
                null_count = df[column].isna().sum()
                if null_count > 0:
                    raise ValidationError(
                        f"Column '{column}' has {null_count} null values "
                        f"but must not be null"
                    )

            # Unique values validation
            elif rule_type == 'unique':
                duplicate_count = df[column].duplicated().sum()
                if duplicate_count > 0:
                    raise ValidationError(
                        f"Column '{column}' has {duplicate_count} duplicate values "
                        f"but must be unique"
                    )

        return True

    def validate_all(self, dataset_path: str) -> Dict[str, bool]:
        """
        Run all validation checks.

        Args:
            dataset_path: Path to CSV dataset

        Returns:
            Dictionary with validation results

        Raises:
            ValidationError if any validation fails
        """
        results = {
            'file_exists': False,
            'schema_valid': False,
            'quality_valid': False,
            'business_rules_valid': False
        }

        # Run validations in order
        results['file_exists'] = self.validate_file_exists(dataset_path)
        results['schema_valid'] = self.validate_dataset_schema(dataset_path)
        results['quality_valid'] = self.validate_data_quality(dataset_path)
        results['business_rules_valid'] = self.validate_business_rules(dataset_path)

        return results


def validate_crew_outputs(required_files: List[str]) -> bool:
    """
    Validate that all required crew outputs exist.

    Args:
        required_files: List of file paths that must exist

    Returns:
        True if all files exist

    Raises:
        ValidationError if any file is missing
    """
    missing_files = []

    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(file_path)

    if missing_files:
        raise ValidationError(
            f"Missing required output files: {missing_files}"
        )

    return True


def validate_artifact_directory(artifact_dir: str = "artifacts") -> bool:
    """
    Validate artifact directory structure exists.

    Args:
        artifact_dir: Path to artifacts directory

    Returns:
        True if directory exists

    Raises:
        ValidationError if directory doesn't exist
    """
    path = Path(artifact_dir)
    if not path.exists():
        raise ValidationError(f"Artifact directory not found: {artifact_dir}")

    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {artifact_dir}")

    return True


def create_validation_report(
    contract_path: str,
    dataset_path: str,
    output_path: str = "artifacts/contract_validation_report.md"
) -> None:
    """
    Create comprehensive validation report.

    Args:
        contract_path: Path to dataset contract
        dataset_path: Path to dataset to validate
        output_path: Where to save the report
    """
    validator = DatasetContractValidator(contract_path)

    report_lines = [
        "# Dataset Contract Validation Report",
        "",
        f"**Dataset**: {dataset_path}",
        f"**Contract**: {contract_path}",
        f"**Generated**: {pd.Timestamp.now()}",
        "",
        "## Validation Results",
        ""
    ]

    try:
        results = validator.validate_all(dataset_path)

        report_lines.append("### ✅ All Validations Passed")
        report_lines.append("")

        for check, passed in results.items():
            status = "✅" if passed else "❌"
            report_lines.append(f"- {status} {check.replace('_', ' ').title()}")

        report_lines.extend([
            "",
            "## Summary",
            "",
            "The dataset successfully passes all contract validations:",
            "- File exists and is not empty",
            "- Schema matches contract specifications",
            "- Data quality meets requirements",
            "- Business rules are satisfied",
            "",
            "**Status**: VALID ✅",
            "",
            "The dataset is ready for feature engineering and model training."
        ])

    except ValidationError as e:
        report_lines.append(f"### ❌ Validation Failed")
        report_lines.append("")
        report_lines.append(f"**Error**: {str(e)}")
        report_lines.append("")
        report_lines.append("**Status**: INVALID ❌")
        report_lines.append("")
        report_lines.append("Please address the validation errors before proceeding.")

    # Write report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Validation report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Validate dataset against contract
    try:
        validator = DatasetContractValidator("artifacts/dataset_contract.json")
        results = validator.validate_all("data/interim/clean_data.csv")
        print("✅ All validations passed!")
        print(results)
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
