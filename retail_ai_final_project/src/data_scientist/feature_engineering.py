"""
Feature Engineering Module
Creates and transforms features for machine learning models.
"""
import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineering transformer.
    Prevents data leakage by fitting on training data only.
    """

    def __init__(self):
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.date_columns = []
        self.quantile_bins = {}
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit the feature engineer on training data.

        Args:
            X: Training DataFrame
            y: Target (unused, for sklearn compatibility)

        Returns:
            self
        """
        logger.info("Fitting feature engineer on training data")
        X = X.copy()

        # Identify column types
        self.numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.date_columns = [col for col in X.columns if 'date' in col.lower()]

        logger.info(f"Found {len(self.numeric_cols)} numeric and {len(self.categorical_cols)} categorical columns")

        # Fit label encoders on categorical variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Fitted label encoder for: {col}")

        # Fit quantile bins on numeric columns (FIX: fit on training data only)
        for col in self.numeric_cols[:2]:  # Limit to first 2 numeric columns
            try:
                _, bins = pd.qcut(X[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                 duplicates='drop', retbins=True)
                self.quantile_bins[col] = bins
                logger.info(f"Fitted quantile bins for: {col}")
            except Exception as e:
                logger.warning(f"Could not create bins for {col}: {e}")

        self.fitted = True
        logger.info("Feature engineer fitting completed")
        return self

    def transform(self, X):
        """
        Transform data using fitted parameters.

        Args:
            X: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        logger.info("Transforming features")
        df_features = X.copy()

        # 1. Encode Categorical Variables using fitted encoders
        for col in self.categorical_cols:
            if col in df_features.columns:
                le = self.label_encoders[col]
                # Handle unseen categories
                df_features[f'{col}_encoded'] = df_features[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                logger.info(f"Encoded categorical feature: {col}")

        # 2. Create Interaction Features
        if len(self.numeric_cols) >= 2:
            col1, col2 = self.numeric_cols[0], self.numeric_cols[1]
            if col1 in df_features.columns and col2 in df_features.columns:
                df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
                logger.info(f"Created interaction feature: {col1}_x_{col2}")

        # 3. Polynomial Features
        for col in self.numeric_cols[:3]:
            if col in df_features.columns:
                df_features[f'{col}_squared'] = df_features[col] ** 2
                logger.info(f"Created polynomial feature: {col}_squared")

        # 4. Binning Numeric Features using fitted bins
        for col, bins in self.quantile_bins.items():
            if col in df_features.columns:
                df_features[f'{col}_binned'] = pd.cut(
                    df_features[col],
                    bins=bins,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],
                    include_lowest=True,
                    duplicates='drop'
                )
                # Encode binned features
                df_features[f'{col}_binned_encoded'] = LabelEncoder().fit_transform(
                    df_features[f'{col}_binned'].astype(str)
                )
                logger.info(f"Created binned feature: {col}_binned")

        # 5. Statistical Aggregations
        available_numeric = [col for col in self.numeric_cols if col in df_features.columns]
        if len(available_numeric) > 2:
            df_features['numeric_mean'] = df_features[available_numeric].mean(axis=1)
            df_features['numeric_std'] = df_features[available_numeric].std(axis=1)
            logger.info("Created statistical aggregation features")

        # 6. Date Features
        for col in self.date_columns:
            if col in df_features.columns and df_features[col].dtype == 'datetime64[ns]':
                df_features[f'{col}_year'] = df_features[col].dt.year
                df_features[f'{col}_month'] = df_features[col].dt.month
                df_features[f'{col}_day'] = df_features[col].dt.day
                df_features[f'{col}_dayofweek'] = df_features[col].dt.dayofweek
                logger.info(f"Created date features from: {col}")

        # Drop original categorical columns (keep encoded versions)
        df_features = df_features.drop(columns=self.categorical_cols, errors='ignore')

        # Drop binned categorical columns (keep encoded versions)
        binned_cols = [col for col in df_features.columns
                      if col.endswith('_binned') and not col.endswith('_encoded')]
        df_features = df_features.drop(columns=binned_cols, errors='ignore')

        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        logger.info(f"Total features: {len(df_features.columns)}")

        return df_features

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def save(self, output_path: str = "artifacts/feature_engineer.pkl"):
        """Save the fitted feature engineer."""
        import pickle
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved feature engineer to {output_path}")

    @staticmethod
    def load(input_path: str = "artifacts/feature_engineer.pkl"):
        """Load a fitted feature engineer."""
        import pickle
        with open(input_path, 'rb') as f:
            feature_engineer = pickle.load(f)
        logger.info(f"Loaded feature engineer from {input_path}")
        return feature_engineer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from cleaned data.
    DEPRECATED: Use FeatureEngineer class for proper train/test handling.

    This function is kept for backward compatibility but should not be used
    in production as it can cause data leakage.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.warning("Using deprecated engineer_features function. "
                   "Use FeatureEngineer class to prevent data leakage.")

    fe = FeatureEngineer()
    df_features = fe.fit_transform(df)

    # Save the feature engineer for later use
    fe.save()

    # Save feature names
    save_feature_names(df_features.columns.tolist())

    return df_features


def save_features(df: pd.DataFrame, output_path: str = "data/processed/features.csv") -> None:
    """
    Save engineered features to CSV file.

    Args:
        df: DataFrame with engineered features
        output_path: Path to save the features
    """
    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")

        # Also save feature names
        save_feature_names(df.columns.tolist())
    except Exception as e:
        logger.error(f"Error saving features: {str(e)}")
        raise


def save_feature_names(feature_names: list, output_path: str = "artifacts/feature_names.json") -> None:
    """
    Save feature names as JSON.

    Args:
        feature_names: List of feature names
        output_path: Path to save feature names
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({"features": feature_names}, f, indent=2)

        logger.info(f"Saved feature names to {output_path}")
    except Exception as e:
        logger.error(f"Error saving feature names: {str(e)}")
        raise


def main():
    """Main execution function for feature engineering."""
    try:
        # Load cleaned data
        df = pd.read_csv("data/interim/clean_data.csv")
        logger.info(f"Loaded cleaned data: {df.shape}")

        # Engineer features using the class-based approach
        df_features = engineer_features(df)

        # Save features
        save_features(df_features)

        logger.info("Feature engineering completed successfully")

    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
