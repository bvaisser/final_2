"""
Model Evaluation Module
Evaluates trained models and generates comprehensive reports.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_model(model_path: str = "artifacts/model.pkl"):
    """Load trained model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance with comprehensive metrics.
    Automatically detects if target is continuous (regression) or categorical (classification).

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance")

    # Make predictions
    y_pred = model.predict(X_test)

    # Detect problem type: regression if target is numeric with many unique values
    is_regression = pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 50

    if is_regression:
        # Calculate regression metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        logger.info(f"Regression metrics: {metrics}")
    else:
        # Calculate classification metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        logger.info(f"Classification metrics: {metrics}")

    return metrics


def generate_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "artifacts/evaluation_report.md"
) -> None:
    """
    Generate comprehensive evaluation report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        output_path: Path to save the report
    """
    logger.info("Generating evaluation report")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = evaluate_model(model, X_test, y_test)

    # Detect problem type
    is_regression = pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 50

    if not is_regression:
        # Generate confusion matrix plot for classification
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('artifacts/confusion_matrix.png')
        plt.close()
        logger.info("Saved confusion matrix plot")

        # Generate ROC curve for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig('artifacts/roc_curve.png')
                plt.close()
                logger.info("Saved ROC curve plot")
            except Exception as e:
                logger.warning(f"Could not generate ROC curve: {e}")
    else:
        # Generate residual plot for regression
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig('artifacts/residual_plot.png')
        plt.close()
        logger.info("Saved residual plot")

        # Generate actual vs predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.savefig('artifacts/actual_vs_predicted.png')
        plt.close()
        logger.info("Saved actual vs predicted plot")

    # Generate feature importance plot (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('artifacts/feature_importance.png')
        plt.close()
        logger.info("Saved feature importance plot")

    # Create markdown report
    report = f"""# Model Evaluation Report

## Model Overview
- **Model Type:** {type(model).__name__}
- **Problem Type:** {'Regression' if is_regression else 'Classification'}
- **Test Set Size:** {len(X_test)} samples
- **Number of Features:** {X_test.shape[1]}

## Performance Metrics

| Metric | Score |
|--------|-------|
"""

    if is_regression:
        report += f"""| MSE | {metrics['mse']:.4f} |
| RMSE | {metrics['rmse']:.4f} |
| MAE | {metrics['mae']:.4f} |
| R² | {metrics['r2']:.4f} |
"""
        report += """
## Residual Plot

![Residual Plot](residual_plot.png)

## Actual vs Predicted

![Actual vs Predicted](actual_vs_predicted.png)
"""
    else:
        report += f"""| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |
"""
        if 'roc_auc' in metrics:
            report += f"| ROC AUC | {metrics['roc_auc']:.4f} |\n"
        
        report += f"""
## Classification Report

```
{classification_report(y_test, y_pred)}
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)
"""
        if len(np.unique(y_test)) == 2 and Path('artifacts/roc_curve.png').exists():
            report += """
## ROC Curve

![ROC Curve](roc_curve.png)
"""

    if hasattr(model, 'feature_importances_'):
        report += """
## Feature Importance

![Feature Importance](feature_importance.png)

### Top 10 Most Important Features

"""
        top_features = feature_importance.head(10)
        for idx, row in top_features.iterrows():
            report += f"- **{row['feature']}**: {row['importance']:.4f}\n"

    report += """
## Recommendations

1. Review misclassified samples to understand model limitations
2. Consider ensemble methods if performance needs improvement
3. Monitor model performance on production data
4. Retrain periodically with new data

## Next Steps

1. Deploy model to production environment
2. Set up monitoring and alerting
3. Collect feedback for model iteration
4. Plan for model versioning and updates
"""

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report saved to {output_path}")


def generate_model_card(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "artifacts/model_card.md"
) -> None:
    """
    Generate a model card documenting the model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        output_path: Path to save the model card
    """
    logger.info("Generating model card")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(model, X_test, y_test)
    
    # Detect problem type
    is_regression = pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 50

    # Build metrics section
    if is_regression:
        metrics_section = f"""- MSE: {metrics['mse']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- R²: {metrics['r2']:.4f}

## Performance

### Quantitative Analysis
- **Mean Squared Error (MSE):** {metrics['mse']:.4f}
- **Root Mean Squared Error (RMSE):** {metrics['rmse']:.4f}
- **Mean Absolute Error (MAE):** {metrics['mae']:.4f}
- **R² Score:** {metrics['r2']:.4f}
"""
    else:
        metrics_section = f"""- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}

## Performance

### Quantitative Analysis
- **Overall Accuracy:** {metrics['accuracy']:.4f}
- **Weighted Precision:** {metrics['precision']:.4f}
- **Weighted Recall:** {metrics['recall']:.4f}
"""

    model_card = f"""# Model Card - Retail AI Model

## Model Details

**Model Type:** {type(model).__name__}
**Version:** 1.0
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Framework:** scikit-learn

## Intended Use

**Primary Use Cases:**
- Retail prediction and classification tasks
- Business intelligence and decision support
- Automated data analysis

**Out-of-Scope Uses:**
- Critical decision-making without human oversight
- Applications outside retail domain
- Real-time inference without validation

## Training Data

**Dataset:** Retail AI Training Dataset
**Size:** Training data from Kaggle
**Features:** {X_test.shape[1]} engineered features
**Preprocessing:** Data cleaning, feature engineering, encoding

## Evaluation Data

**Test Set Size:** {len(X_test)} samples
**Problem Type:** {'Regression' if is_regression else 'Classification'}
**Evaluation Metrics:**
{metrics_section}
### Limitations
- Model trained on historical data may not capture recent trends
- Performance may degrade on significantly different data distributions
- Requires periodic retraining with fresh data

## Ethical Considerations

**Potential Biases:**
- Model reflects patterns in training data, which may contain historical biases
- Should be regularly audited for fairness across different segments

**Recommendations:**
- Monitor for bias in predictions across different customer segments
- Implement human oversight for critical business decisions
- Regular fairness audits and model updates

## Maintenance

**Monitoring:**
- Track prediction accuracy over time
- Monitor for data drift
- Log prediction distributions

**Update Frequency:**
- Recommended retraining: Monthly or quarterly
- Update when performance degrades below threshold

## Contact

For questions or issues, contact the ML team.

## References

- Project repository: [Link to repository]
- Training pipeline: `crew/crew_flow.py`
- Evaluation metrics: `artifacts/evaluation_report.md`
"""

    with open(output_path, 'w') as f:
        f.write(model_card)

    logger.info(f"Model card saved to {output_path}")


def main():
    """Main execution function for model evaluation."""
    try:
        # Load model
        model = load_model()

        # Load feature-engineered data
        df = pd.read_csv("data/processed/features.csv")
        logger.info(f"Loaded features: {df.shape}")

        # Prepare data (assuming last column is target)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data (same as training)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Generate reports
        generate_evaluation_report(model, X_test, y_test)
        generate_model_card(model, X_test, y_test)

        logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
