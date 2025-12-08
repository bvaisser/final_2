#!/usr/bin/env python
"""
Retail AI Final Project - Main Entry Point
Orchestrates the entire ML pipeline with CLI interface.

Usage:
    python main.py --help                           # Show all options
    python main.py --pipeline full                  # Run entire pipeline
    python main.py --pipeline data                  # Run data analysis only
    python main.py --pipeline model                 # Run model training only
    python main.py --app streamlit                  # Launch Streamlit app
    python main.py --app flask                      # Launch Flask API
    python main.py --crew                           # Run CrewAI workflow
"""
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger
from src.data_analyst.ingest_and_clean import load_data, clean_data, save_clean_data
from src.data_analyst.eda import generate_eda_report, generate_insights
from src.data_analyst.build_contract import create_dataset_contract
from src.data_scientist.feature_engineering import engineer_features, save_features
from src.data_scientist.train_models import prepare_data, train_model, save_model
from src.data_scientist.evaluate_models import generate_evaluation_report, generate_model_card
import pandas as pd
import json

logger = get_logger(__name__)


class RetailAIPipeline:
    """Main pipeline orchestrator for the Retail AI project."""

    def __init__(self):
        self.raw_data_path = "data/raw/Coffe_sales.csv"
        self.clean_data_path = "data/interim/clean_data.csv"
        self.features_path = "data/processed/features.csv"
        self.model_path = "artifacts/model.pkl"
        self.contract_path = "artifacts/dataset_contract.json"

    def run_data_pipeline(self) -> pd.DataFrame:
        """
        Execute the data analysis pipeline.

        Returns:
            Cleaned DataFrame
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA ANALYSIS PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: Load data
            print("\n📊 Step 1/5: Loading raw data...")
            df_raw = load_data(self.raw_data_path)
            print(f"  ✅ Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

            # Step 2: Clean data
            print("\n🧹 Step 2/5: Cleaning data...")
            df_clean = clean_data(df_raw)
            save_clean_data(df_clean, self.clean_data_path)
            print(f"  ✅ Cleaned data saved: {df_clean.shape}")

            # Step 3: Generate EDA
            print("\n📈 Step 3/5: Generating EDA report...")
            generate_eda_report(df_clean)
            print("  ✅ EDA report created: artifacts/eda_report.html")

            # Step 4: Generate insights
            print("\n💡 Step 4/5: Extracting insights...")
            generate_insights(df_clean)
            print("  ✅ Insights saved: artifacts/insights.md")

            # Step 5: Create contract
            print("\n📋 Step 5/5: Creating dataset contract...")
            create_dataset_contract(df_clean)
            print("  ✅ Contract created: artifacts/dataset_contract.json")

            logger.info("Data analysis pipeline completed successfully")
            return df_clean

        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            raise

    def run_model_pipeline(self, df_clean: Optional[pd.DataFrame] = None) -> None:
        """
        Execute the model training pipeline.

        Args:
            df_clean: Optional cleaned DataFrame (if None, loads from file)
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        try:
            # Load clean data if not provided
            if df_clean is None:
                print("\n📂 Loading cleaned data from disk...")
                df_clean = pd.read_csv(self.clean_data_path)
                print(f"  ✅ Loaded {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")

            # Step 1: Feature engineering
            print("\n🔧 Step 1/5: Engineering features...")
            df_features = engineer_features(df_clean)
            save_features(df_features, self.features_path)
            print(f"  ✅ Features engineered: {df_features.shape}")

            # Step 2: Load contract to get target column
            print("\n🎯 Step 2/5: Preparing train/test split...")
            with open(self.contract_path, 'r') as f:
                contract = json.load(f)
                target_column = contract.get('target_column', df_features.columns[-1])

            X, y = prepare_data(df_features, target_column=target_column)
            print(f"  ✅ Data prepared: X={X.shape}, y={y.shape}")

            # Step 3: Train model
            print("\n🤖 Step 3/5: Training model (Random Forest)...")
            model = train_model(X, y, model_type='random_forest')
            save_model(model, self.model_path)
            print("  ✅ Model trained and saved")

            # Step 4: Evaluate model
            print("\n📊 Step 4/5: Evaluating model...")
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            generate_evaluation_report(model, X_test, y_test)
            print("  ✅ Evaluation report created: artifacts/evaluation_report.md")

            # Step 5: Generate model card
            print("\n📝 Step 5/5: Generating model card...")
            generate_model_card(model, X_test, y_test)
            print("  ✅ Model card created: artifacts/model_card.md")

            logger.info("Model training pipeline completed successfully")

        except Exception as e:
            logger.error(f"Model pipeline failed: {str(e)}")
            raise

    def run_full_pipeline(self) -> None:
        """Execute the complete end-to-end pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING FULL PIPELINE (DATA + MODEL)")
        logger.info("=" * 60)

        try:
            # Run data pipeline
            df_clean = self.run_data_pipeline()

            print("\n" + "=" * 60)
            print("Data pipeline complete. Starting model pipeline...")
            print("=" * 60)

            # Run model pipeline
            self.run_model_pipeline(df_clean)

            # Print summary
            self._print_summary()

        except Exception as e:
            logger.error(f"Full pipeline failed: {str(e)}")
            raise

    def _print_summary(self) -> None:
        """Print pipeline completion summary."""
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📁 Generated Artifacts:")
        print(f"  - Cleaned Data: {self.clean_data_path}")
        print(f"  - Features: {self.features_path}")
        print(f"  - Model: {self.model_path}")
        print("  - EDA Report: artifacts/eda_report.html")
        print("  - Insights: artifacts/insights.md")
        print("  - Dataset Contract: artifacts/dataset_contract.json")
        print("  - Evaluation Report: artifacts/evaluation_report.md")
        print("  - Model Card: artifacts/model_card.md")

        print("\n🚀 Next Steps:")
        print("  1. Review artifacts in the artifacts/ directory")
        print("  2. Launch the Streamlit app:")
        print("     python main.py --app streamlit")
        print("  3. Or launch the Flask API:")
        print("     python main.py --app flask")


def launch_streamlit() -> None:
    """Launch the Streamlit web application."""
    print("\n🚀 Launching Streamlit App...")
    print("=" * 60)
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"

    if not app_path.exists():
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)

    logger.info(f"Starting Streamlit app from {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)])


def launch_flask() -> None:
    """Launch the Flask API server."""
    print("\n🚀 Launching Flask API...")
    print("=" * 60)
    app_path = PROJECT_ROOT / "app" / "flask_app.py"

    if not app_path.exists():
        logger.error(f"Flask app not found at {app_path}")
        sys.exit(1)

    logger.info(f"Starting Flask app from {app_path}")
    subprocess.run(["python", str(app_path)])


def launch_crew() -> None:
    """Launch the CrewAI workflow."""
    print("\n🤖 Launching CrewAI Workflow...")
    print("=" * 60)
    crew_path = PROJECT_ROOT / "crew" / "crew_flow.py"

    if not crew_path.exists():
        logger.error(f"Crew flow not found at {crew_path}")
        sys.exit(1)

    logger.info(f"Starting CrewAI flow from {crew_path}")
    subprocess.run(["python", str(crew_path)])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Retail AI Final Project - ML Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the complete pipeline
  python main.py --pipeline full

  # Run only data analysis
  python main.py --pipeline data

  # Run only model training (requires clean data)
  python main.py --pipeline model

  # Launch Streamlit web app
  python main.py --app streamlit

  # Launch Flask API
  python main.py --app flask

  # Run CrewAI workflow
  python main.py --crew
        """
    )

    # Pipeline arguments
    parser.add_argument(
        "--pipeline",
        choices=["full", "data", "model"],
        help="Run a specific pipeline stage"
    )

    # App arguments
    parser.add_argument(
        "--app",
        choices=["streamlit", "flask"],
        help="Launch a specific application"
    )

    # Crew argument
    parser.add_argument(
        "--crew",
        action="store_true",
        help="Run the CrewAI workflow"
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="Retail AI Pipeline v1.0.0"
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    try:
        # Change to project root directory
        import os
        os.chdir(PROJECT_ROOT)

        # Execute based on arguments
        if args.pipeline:
            pipeline = RetailAIPipeline()

            if args.pipeline == "full":
                pipeline.run_full_pipeline()
            elif args.pipeline == "data":
                pipeline.run_data_pipeline()
                print("\n✅ Data pipeline complete!")
                print("Next: python main.py --pipeline model")
            elif args.pipeline == "model":
                pipeline.run_model_pipeline()
                print("\n✅ Model pipeline complete!")

        elif args.app:
            if args.app == "streamlit":
                launch_streamlit()
            elif args.app == "flask":
                launch_flask()

        elif args.crew:
            launch_crew()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
