"""
Model Training Module
Trains multiple machine learning models and selects the best one.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_data(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training by separating features and target.

    Args:
        df: DataFrame with features
        target_column: Name of the target column (if None, assumes last column)

    Returns:
        Tuple of (X, y) - features and target
    """
    if target_column is None:
        # Assume last column is the target
        target_column = df.columns[-1]
        logger.warning(f"No target column specified. Using last column: {target_column}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Any:
    """
    Train a machine learning model with hyperparameter tuning.
    Automatically detects if target is continuous (regression) or categorical (classification).

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model to train ('random_forest', 'gradient_boosting', 'logistic_regression')

    Returns:
        Trained model
    """
    # Validate model type early
    valid_model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid options: {', '.join(valid_model_types)}"
        )

    # Detect problem type: regression if target is numeric with many unique values
    is_regression = pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 50
    
    if is_regression:
        logger.info(f"Detected regression problem (target has {len(y.unique())} unique values)")
        problem_type = 'regression'
    else:
        logger.info(f"Detected classification problem (target has {len(y.unique())} unique values)")
        problem_type = 'classification'
    
    logger.info(f"Training {model_type} model for {problem_type}")

    # Split data with stratification (try stratified first, fall back to non-stratified)
    stratify_param = None
    if not is_regression and len(y.unique()) < 50:
        # Check if we have enough samples per class for stratification
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify_param = y
        else:
            logger.warning(f"Skipping stratification: minimum class count is {min_class_count} (need >= 2)")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    except ValueError as e:
        # If stratification fails, retry without it
        logger.warning(f"Stratification failed: {e}. Retrying without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define models and hyperparameter grids
    if model_type == 'random_forest':
        if is_regression:
            model = RandomForestRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        else:
            model = RandomForestClassifier(random_state=42)
            scoring = 'accuracy'
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'gradient_boosting':
        if is_regression:
            model = GradientBoostingRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        else:
            model = GradientBoostingClassifier(random_state=42)
            scoring = 'accuracy'
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'logistic_regression':
        if is_regression:
            model = LinearRegression()
            scoring = 'neg_mean_squared_error'
            param_grid = {}  # Linear regression doesn't have hyperparameters to tune
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            scoring = 'accuracy'
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }

    # Perform GridSearchCV
    logger.info("Starting hyperparameter tuning with GridSearchCV")
    if param_grid:
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    else:
        # No hyperparameters to tune, just fit the model
        model.fit(X_train, y_train)
        best_model = model
        logger.info("Model trained (no hyperparameter tuning needed)")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    if is_regression:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Test MSE: {mse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test R²: {r2:.4f}")
    else:
        test_accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

    return best_model


def train_multiple_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train multiple models and compare their performance.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Dictionary of trained models
    """
    logger.info("Training multiple models for comparison")

    models = {}
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']

    for model_type in model_types:
        try:
            model = train_model(X, y, model_type)
            models[model_type] = model
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {str(e)}")

    return models


def save_model(model: Any, output_path: str = "artifacts/model.pkl") -> None:
    """
    Save trained model to pickle file.

    Args:
        model: Trained model
        output_path: Path to save the model
    """
    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Saved model to {output_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def main():
    """Main execution function for model training."""
    try:
        # Load feature-engineered data
        df = pd.read_csv("data/processed/features.csv")
        logger.info(f"Loaded features: {df.shape}")

        # Prepare data (customize target_column based on your dataset)
        # For demonstration, assuming last column is target
        X, y = prepare_data(df)

        # Train best model (Random Forest by default)
        model = train_model(X, y, model_type='random_forest')

        # Optionally train multiple models
        # all_models = train_multiple_models(X, y)

        # Save model
        save_model(model)

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
