"""
CrewAI Tools for Data Analysis and Model Training
Wraps Python functions as tools that agents can use.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from crewai.tools import tool
from src.data_analyst.ingest_and_clean import load_data, clean_data, save_clean_data
from src.data_analyst.eda import perform_eda, generate_eda_report, generate_insights
from src.data_analyst.build_contract import create_dataset_contract
from src.data_scientist.feature_engineering import FeatureEngineer, save_features
from src.data_scientist.train_models import prepare_data, train_model, save_model
from src.data_scientist.evaluate_models import (
    load_model, evaluate_model, generate_evaluation_report, generate_model_card
)


# ============================================================
# DATA ANALYST TOOLS
# ============================================================

@tool("Load Dataset")
def load_dataset_tool(file_path: str = "data/raw/Coffe_sales.csv") -> str:
    """
    Load dataset from CSV file and return summary information.

    Args:
        file_path: Path to the CSV file to load

    Returns:
        Summary of loaded dataset
    """
    try:
        df = load_data(file_path)
        return f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


@tool("Clean Dataset")
def clean_dataset_tool() -> str:
    """
    Clean the raw dataset by handling missing values and duplicates.
    Saves cleaned data to data/interim/clean_data.csv

    Returns:
        Summary of cleaning operations
    """
    try:
        df_raw = load_data("data/raw/Coffe_sales.csv")
        df_clean = clean_data(df_raw)
        save_clean_data(df_clean)
        return (f"Successfully cleaned dataset: {df_clean.shape[0]} rows, "
                f"{df_clean.shape[1]} columns. Saved to data/interim/clean_data.csv")
    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"


@tool("Generate EDA Report")
def generate_eda_tool() -> str:
    """
    Generate comprehensive EDA report with visualizations.
    Creates artifacts/eda_report.html

    Returns:
        Confirmation message
    """
    try:
        df = pd.read_csv("data/interim/clean_data.csv")
        generate_eda_report(df)
        return "Successfully generated EDA report at artifacts/eda_report.html"
    except Exception as e:
        return f"Error generating EDA report: {str(e)}"


@tool("Generate Insights")
def generate_insights_tool() -> str:
    """
    Generate business insights from EDA.
    Creates artifacts/insights.md

    Returns:
        Confirmation message
    """
    try:
        df = pd.read_csv("data/interim/clean_data.csv")
        generate_insights(df)
        return "Successfully generated insights at artifacts/insights.md"
    except Exception as e:
        return f"Error generating insights: {str(e)}"


@tool("Create Dataset Contract")
def create_contract_tool() -> str:
    """
    Create formal dataset contract with schema and constraints.
    Creates artifacts/dataset_contract.json

    Returns:
        Confirmation message
    """
    try:
        df = pd.read_csv("data/interim/clean_data.csv")

        # Detect target column (last column or 'sales' if exists)
        target_column = 'sales' if 'sales' in df.columns else df.columns[-1]

        contract = create_dataset_contract(df)
        # Add target column to contract
        contract['target_column'] = target_column

        # Save updated contract
        import json
        with open("artifacts/dataset_contract.json", 'w') as f:
            json.dump(contract, f, indent=2)

        return (f"Successfully created dataset contract at artifacts/dataset_contract.json. "
                f"Target column: {target_column}")
    except Exception as e:
        return f"Error creating contract: {str(e)}"


# ============================================================
# DATA SCIENTIST TOOLS
# ============================================================

@tool("Engineer Features")
def engineer_features_tool() -> str:
    """
    Engineer features from cleaned dataset using FeatureEngineer class.
    Saves features to data/processed/features.csv

    Returns:
        Confirmation message
    """
    try:
        df_clean = pd.read_csv("data/interim/clean_data.csv")

        # Use FeatureEngineer class to prevent data leakage
        fe = FeatureEngineer()
        df_features = fe.fit_transform(df_clean)

        # Save feature engineer and features
        fe.save("artifacts/feature_engineer.pkl")
        save_features(df_features)

        return (f"Successfully engineered features: {df_features.shape[0]} rows, "
                f"{df_features.shape[1]} features. Saved to data/processed/features.csv")
    except Exception as e:
        return f"Error engineering features: {str(e)}"


@tool("Train Model")
def train_model_tool(model_type: str = "random_forest") -> str:
    """
    Train machine learning model on engineered features.
    Saves model to artifacts/model.pkl

    Args:
        model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')

    Returns:
        Confirmation message with performance metrics
    """
    try:
        df_features = pd.read_csv("data/processed/features.csv")

        # Read target column from contract
        import json
        with open("artifacts/dataset_contract.json", 'r') as f:
            contract = json.load(f)
            target_column = contract.get('target_column', df_features.columns[-1])

        # Prepare data
        X, y = prepare_data(df_features, target_column=target_column)

        # Train model
        model = train_model(X, y, model_type=model_type)

        # Save model
        save_model(model)

        return f"Successfully trained {model_type} model. Saved to artifacts/model.pkl"
    except Exception as e:
        return f"Error training model: {str(e)}"


@tool("Evaluate Model")
def evaluate_model_tool() -> str:
    """
    Evaluate trained model and generate comprehensive reports.
    Creates artifacts/evaluation_report.md and artifacts/model_card.md

    Returns:
        Confirmation message
    """
    try:
        # Load model and data
        model = load_model()
        df_features = pd.read_csv("data/processed/features.csv")

        # Read target column from contract
        import json
        with open("artifacts/dataset_contract.json", 'r') as f:
            contract = json.load(f)
            target_column = contract.get('target_column', df_features.columns[-1])

        # Prepare test data
        from sklearn.model_selection import train_test_split
        X = df_features.drop(columns=[target_column])
        y = df_features[target_column]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Generate reports
        generate_evaluation_report(model, X_test, y_test)
        generate_model_card(model, X_test, y_test)

        # Get metrics
        metrics = evaluate_model(model, X_test, y_test)

        return (f"Successfully evaluated model. "
                f"Reports saved to artifacts/evaluation_report.md and artifacts/model_card.md. "
                f"Metrics: {metrics}")
    except Exception as e:
        return f"Error evaluating model: {str(e)}"
