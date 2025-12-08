"""
Exploratory Data Analysis Module
Performs comprehensive EDA and generates reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def perform_eda(df: pd.DataFrame) -> dict:
    """
    Perform exploratory data analysis on the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing EDA results
    """
    logger.info("Starting exploratory data analysis")

    eda_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': df.describe().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns},
    }

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    eda_results['numeric_columns'] = numeric_cols
    eda_results['categorical_columns'] = categorical_cols

    logger.info(f"EDA completed for {df.shape[0]} rows and {df.shape[1]} columns")

    return eda_results


def generate_eda_report(df: pd.DataFrame, output_path: str = "artifacts/eda_report.html") -> None:
    """
    Generate HTML EDA report with visualizations.

    Args:
        df: DataFrame to analyze
        output_path: Path to save the HTML report
    """
    logger.info("Generating EDA report")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Create figure directory
    fig_dir = Path("artifacts/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Distribution plots for numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(fig_dir / f'{col}_distribution.png')
        plt.close()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(fig_dir / 'correlation_heatmap.png')
        plt.close()

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report - Retail AI Project</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Exploratory Data Analysis Report</h1>

        <h2>Dataset Overview</h2>
        <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>

        <h2>Data Types</h2>
        <table>
            <tr><th>Column</th><th>Data Type</th><th>Non-Null Count</th><th>Unique Values</th></tr>
            {''.join([f'<tr><td>{col}</td><td>{df[col].dtype}</td><td>{df[col].count()}</td><td>{df[col].nunique()}</td></tr>' for col in df.columns])}
        </table>

        <h2>Statistical Summary</h2>
        {df.describe().to_html()}

        <h2>Missing Values</h2>
        <table>
            <tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>
            {''.join([f'<tr><td>{col}</td><td>{df[col].isnull().sum()}</td><td>{df[col].isnull().sum() / len(df) * 100:.2f}%</td></tr>' for col in df.columns])}
        </table>

        <h2>Visualizations</h2>
        <h3>Correlation Heatmap</h3>
        <img src="figures/correlation_heatmap.png" alt="Correlation Heatmap">

        <h3>Feature Distributions</h3>
        {''.join([f'<img src="figures/{col}_distribution.png" alt="{col} Distribution">' for col in numeric_cols[:5]])}
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"EDA report saved to {output_path}")


def generate_insights(df: pd.DataFrame, output_path: str = "artifacts/insights.md") -> None:
    """
    Generate insights document from EDA.

    Args:
        df: DataFrame to analyze
        output_path: Path to save the insights markdown file
    """
    logger.info("Generating insights document")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    eda_results = perform_eda(df)

    insights = f"""# Data Insights - Retail AI Project

## Dataset Overview
- **Total Records:** {eda_results['shape'][0]:,}
- **Total Features:** {eda_results['shape'][1]}
- **Numeric Features:** {len(eda_results['numeric_columns'])}
- **Categorical Features:** {len(eda_results['categorical_columns'])}

## Data Quality
### Missing Values
"""

    # Add missing value information
    missing_info = [(col, count) for col, count in eda_results['missing_values'].items() if count > 0]
    if missing_info:
        insights += "\n"
        for col, count in missing_info:
            pct = (count / eda_results['shape'][0]) * 100
            insights += f"- **{col}**: {count} missing ({pct:.2f}%)\n"
    else:
        insights += "\nNo missing values detected.\n"

    insights += f"""
## Feature Types

### Numeric Features
{chr(10).join([f'- {col}' for col in eda_results['numeric_columns']])}

### Categorical Features
{chr(10).join([f'- {col}' for col in eda_results['categorical_columns']])}

## Key Observations
1. Dataset contains {eda_results['shape'][0]:,} records across {eda_results['shape'][1]} features
2. Data quality is {'good' if len(missing_info) == 0 else 'acceptable with some missing values'}
3. Mix of numeric and categorical features suitable for modeling

## Recommendations
1. Review correlation heatmap for multicollinearity
2. Consider feature engineering for categorical variables
3. Investigate outliers in numeric features
4. Validate business logic for key features

## Next Steps
1. Feature engineering based on insights
2. Model selection and training
3. Hyperparameter tuning
"""

    with open(output_path, 'w') as f:
        f.write(insights)

    logger.info(f"Insights document saved to {output_path}")


def main():
    """Main execution function for EDA."""
    try:
        # Load cleaned data
        df = pd.read_csv("data/interim/clean_data.csv")
        logger.info(f"Loaded cleaned data: {df.shape}")

        # Perform EDA
        eda_results = perform_eda(df)

        # Generate reports
        generate_eda_report(df)
        generate_insights(df)

        logger.info("EDA completed successfully")

    except Exception as e:
        logger.error(f"EDA pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
