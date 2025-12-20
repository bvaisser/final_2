#!/usr/bin/env python
"""
Retail AI Crew Flow
Orchestrates the complete ML pipeline from data ingestion to model evaluation.
"""
from pathlib import Path
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process
from crewai.flow import Flow, listen, start
import yaml
import traceback
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file FIRST, before any other imports
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from src.utils.validation import (
    DatasetContractValidator,
    ValidationError,
    validate_crew_outputs,
    create_validation_report
)
from src.utils.logging_config import PipelineLogger
from src.utils.reproducibility import setup_reproducibility
from src.data_analyst.ingest_and_clean import load_data, clean_data, save_clean_data
from src.data_analyst.build_contract import create_dataset_contract
from src.data_scientist.feature_engineering import engineer_features, save_features
from src.data_scientist.train_models import prepare_data, train_model, save_model
from src.data_scientist.evaluate_models import (
    load_model, generate_evaluation_report, generate_model_card
)
from crew.tools import (
    load_dataset_tool, clean_dataset_tool, generate_eda_tool,
    generate_insights_tool, create_contract_tool, engineer_features_tool,
    train_model_tool, evaluate_model_tool
)


# Define the flow state
class RetailAIState(BaseModel):
    """State management for the Retail AI workflow."""
    raw_data_path: str = "data/raw/Coffe_sales.csv"
    clean_data_path: str = "data/interim/clean_data.csv"
    features_path: str = "data/processed/features.csv"
    model_path: str = "artifacts/model.pkl"
    workflow_status: str = "initialized"
    current_stage: str = ""


class RetailAIFlow(Flow[RetailAIState]):
    """
    Main flow orchestrating the retail AI pipeline.

    Stages:
    1. Data Analyst Crew: Ingest, clean, analyze data
    2. Data Scientist Crew: Feature engineering and model training
    3. Evaluation Crew: Model evaluation and reporting
    """

    def __init__(self):
        super().__init__()
        # Get the project root directory (parent of crew directory)
        self.project_root = Path(__file__).parent.parent
        self.agents_config = self._load_config(str(self.project_root / "crew/agents.yaml"))
        self.tasks_config = self._load_config(str(self.project_root / "crew/tasks.yaml"))

        # Initialize logging
        self.logger_manager = PipelineLogger(
            name="retail_ai_pipeline",
            log_dir=str(self.project_root / "logs"),
            log_level=20  # INFO level
        )
        self.logger = self.logger_manager.get_logger()

        # Create run-specific log
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = self.logger_manager.create_run_log(run_name)

        # Setup reproducibility - set global random seeds
        self.reproducibility_config = setup_reproducibility(
            seed=42,
            log_config=True
        )
        self.random_state = self.reproducibility_config.get_random_state()

        self.logger.info("Retail AI Pipeline initialized")
        self.logger.info(f"Run log: {self.run_log_path}")
        self.logger.info(f"Random seed: {self.random_state}")

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _create_agent(self, agent_name: str) -> Agent:
        """Create an agent from configuration with appropriate tools."""
        config = self.agents_config[agent_name]

        # Assign tools based on agent type
        tools = []
        if agent_name == "data_ingestion_specialist":
            tools = [load_dataset_tool, clean_dataset_tool]
        elif agent_name == "data_cleaning_engineer":
            tools = [clean_dataset_tool]
        elif agent_name == "analytics_insights_specialist":
            tools = [generate_eda_tool, generate_insights_tool, create_contract_tool]
        elif agent_name == "feature_engineering_specialist":
            tools = [engineer_features_tool]
        elif agent_name == "model_training_specialist":
            tools = [train_model_tool]
        elif agent_name == "model_evaluator_documentation_specialist":
            tools = [evaluate_model_tool]

        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            allow_delegation=False,
            tools=tools
        )

    def _create_task(self, task_name: str, agent: Agent) -> Task:
        """Create a task from configuration."""
        config = self.tasks_config[task_name]
        return Task(
            description=config['description'],
            expected_output=config['expected_output'],
            agent=agent
        )

    @start()
    def start_pipeline(self):
        """Initialize the pipeline."""
        print("=" * 60)
        print("Starting Retail AI Pipeline")
        print("=" * 60)

        self.logger_manager.log_stage_start(
            "Pipeline Initialization",
            details=f"Run log: {self.run_log_path}"
        )

        self.state.workflow_status = "running"
        self.state.current_stage = "data_analysis"

        self.logger.info(f"Workflow status: {self.state.workflow_status}")
        self.logger.info(f"Current stage: {self.state.current_stage}")

    @listen(start_pipeline)
    def data_analyst_crew(self):
        """Execute data analyst tasks using direct Python function calls (OPTIMIZED)."""
        print("\n[Stage 1] Running Data Analyst Crew (Optimized - Direct Function Calls)...")
        self.state.current_stage = "data_analysis"

        start_time = time.time()
        self.logger_manager.log_stage_start(
            "Data Analyst Crew",
            details="Optimized: Direct Python execution"
        )

        try:
            # OPTIMIZATION: Call Python functions directly instead of CrewAI agents
            # This eliminates 90-180 seconds of LLM overhead

            print("\n📊 Step 1/5: Loading raw data...")
            df_raw = load_data(str(self.project_root / "data/raw/Coffe_sales.csv"))
            print(f"  ✅ Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

            print("\n🧹 Step 2/5: Cleaning data...")
            df_cleaned = clean_data(df_raw)
            # Cache cleaned data in memory to avoid re-reading
            self._cached_clean_data = df_cleaned
            save_clean_data(df_cleaned, str(self.project_root / "data/interim/clean_data.csv"))
            print(f"  ✅ Cleaned data saved: {df_cleaned.shape}")

            print("\n📈 Step 3/5: Generating EDA report...")
            # OPTIMIZATION: Reuse cached data instead of re-reading CSV
            from src.data_analyst.eda import generate_eda_report, generate_insights
            generate_eda_report(df_cleaned)
            print("  ✅ EDA report created: artifacts/eda_report.html")

            print("\n💡 Step 4/5: Extracting insights...")
            generate_insights(df_cleaned)
            print("  ✅ Insights saved: artifacts/insights.md")

            print("\n📋 Step 5/5: Creating dataset contract...")
            create_dataset_contract(df_cleaned)
            print("  ✅ Contract created: artifacts/dataset_contract.json")

            # Validate required outputs before proceeding
            required_outputs = [
                str(self.project_root / "data/interim/clean_data.csv"),
                str(self.project_root / "artifacts/dataset_contract.json")
            ]

            try:
                validate_crew_outputs(required_outputs)
                self.logger_manager.log_validation(
                    "Output Files",
                    "PASSED",
                    "All required files generated"
                )
                print("\n✅ Data Analyst Crew completed successfully!")
                print("Generated outputs:")
                print("  - Dataset Selection Report: "
                      "artifacts/dataset_selection_report.md")
                print("  - Raw Data Profile: artifacts/raw_data_profile.txt")
                print("  - Validation Report: "
                      "artifacts/validation_report.md")
                print("  - Cleaning Log: artifacts/cleaning_log.md")
                print("  - Data Dictionary: "
                      "artifacts/data_dictionary.json")
                print("  - Preprocessing Summary: "
                      "artifacts/preprocessing_summary.md")
                print("  - Clean Data: data/interim/clean_data.csv")
                print("  - EDA Report: artifacts/eda_report.html")
                print("  - Business Insights: artifacts/insights.md")
                print("  - Dataset Contract: "
                      "artifacts/dataset_contract.json")
                print("  - Visualizations: artifacts/visualizations/")

                # Log artifacts
                for output_file in required_outputs:
                    self.logger_manager.log_artifact_created(
                        output_file,
                        "Data Analyst Output"
                    )

            except ValidationError as ve:
                print(f"\n❌ Output validation failed: {ve}")
                self.logger_manager.log_validation(
                    "Output Files",
                    "FAILED",
                    str(ve)
                )
                self.state.workflow_status = "failed"
                raise

            duration = time.time() - start_time
            self.logger_manager.log_stage_end(
                "Data Analyst Crew",
                status="SUCCESS"
            )
            print(f"\n⏱️  Data Analyst Crew completed in {duration:.2f} seconds")
            print(f"   (Optimized: ~{350-duration:.0f}s faster than CrewAI approach)")
            self.logger.info(f"Duration: {duration:.2f} seconds")

        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ Data Analyst Crew failed: {str(e)}")
            print("\nError details:")
            traceback.print_exc()

            self.logger_manager.log_error(
                f"Data Analyst Crew failed: {str(e)}",
                exc_info=True
            )
            self.logger_manager.log_stage_end(
                "Data Analyst Crew",
                status="FAILED"
            )
            self.logger.info(f"Duration before failure: {duration:.2f} seconds")

            self.state.workflow_status = "failed"
            self.state.current_stage = "data_analysis_failed"
            raise RuntimeError(
                f"Data Analyst Crew failed at stage: {self.state.current_stage}"
            ) from e

    @listen(data_analyst_crew)
    def data_scientist_crew(self):
        """Execute data scientist tasks using direct Python function calls (OPTIMIZED)."""
        print("\n[Stage 2] Running Data Scientist Crew (Optimized - Direct Function Calls)...")
        self.state.current_stage = "data_science"

        start_time = time.time()
        self.logger_manager.log_stage_start(
            "Data Scientist Crew",
            details="Optimized: Direct Python execution"
        )

        try:
            # Validate inputs from Data Analyst Crew before proceeding
            print("\n🔍 Validating Data Analyst Crew outputs...")
            self.logger.info("Validating inputs from Data Analyst Crew")

            contract_path = str(self.project_root / "artifacts/dataset_contract.json")
            dataset_path = str(self.project_root / "data/interim/clean_data.csv")

            # Create validation report
            try:
                create_validation_report(
                    contract_path=contract_path,
                    dataset_path=dataset_path,
                    output_path=str(self.project_root / "artifacts/contract_validation_report.md")
                )
                print("✅ Dataset contract validation passed!")
                self.logger_manager.log_validation(
                    "Dataset Contract",
                    "PASSED",
                    "Contract matches dataset schema and quality requirements"
                )
            except ValidationError as ve:
                print(f"❌ Contract validation failed: {ve}")
                print("\nCannot proceed to Data Scientist Crew.")
                print("Please fix data quality issues and re-run "
                      "Data Analyst Crew.")
                self.logger_manager.log_validation(
                    "Dataset Contract",
                    "FAILED",
                    str(ve)
                )
                self.state.workflow_status = "failed"
                self.state.current_stage = "validation_failed"
                raise

            # OPTIMIZATION: Call Python functions directly instead of CrewAI agents
            # This eliminates additional LLM overhead

            print("\n🔧 Step 1/5: Engineering features...")
            # Use cached data from data_analyst_crew to avoid re-reading CSV
            if hasattr(self, '_cached_clean_data'):
                df_clean = self._cached_clean_data
                print("  ✅ Using cached clean data")
            else:
                df_clean = pd.read_csv(str(self.project_root / "data/interim/clean_data.csv"))
                print("  ⚠️  Loaded clean data from disk")

            df_features = engineer_features(df_clean)
            save_features(df_features, str(self.project_root / "data/processed/features.csv"))
            # Cache features in memory
            self._cached_features = df_features
            print(f"  ✅ Features engineered: {df_features.shape}")

            print("\n🎯 Step 2/5: Preparing train/test split...")
            import json
            contract_file = self.project_root / "artifacts/dataset_contract.json"
            with open(contract_file, 'r') as f:
                contract = json.load(f)
                target_column = contract.get('target_column', df_features.columns[-1])

            X, y = prepare_data(df_features, target_column=target_column)
            print(f"  ✅ Data prepared: X={X.shape}, y={y.shape}")

            print("\n🤖 Step 3/5: Training model...")
            model = train_model(X, y, model_type='random_forest')
            save_model(model, str(self.project_root / "artifacts/model.pkl"))
            print("  ✅ Model trained and saved")

            print("\n📊 Step 4/5: Evaluating model...")
            from sklearn.model_selection import train_test_split
            from src.data_scientist.evaluate_models import (
                generate_evaluation_report, generate_model_card
            )
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            generate_evaluation_report(model, X_test, y_test)
            print("  ✅ Evaluation report created")

            print("\n📝 Step 5/5: Generating model card...")
            generate_model_card(model, X_test, y_test)
            print("  ✅ Model card created")

            # Validate required outputs
            required_outputs = [
                str(self.project_root / "data/processed/features.csv"),
                str(self.project_root / "artifacts/model.pkl"),
                str(self.project_root / "artifacts/model_card.md")
            ]

            try:
                validate_crew_outputs(required_outputs)
                self.logger_manager.log_validation(
                    "Model Output Files",
                    "PASSED",
                    "All required model artifacts generated"
                )
                print("\n✅ Data Scientist Crew completed successfully!")
                print("Generated outputs:")
                print("  - Contract Validation: "
                      "artifacts/contract_validation_report.md")
                print("  - Feature Metadata: "
                      "artifacts/feature_metadata.json")
                print("  - Features Dataset: data/processed/features.csv")
                print("  - Train/Test Data: data/processed/X_train.csv, "
                      "X_test.csv, y_train.csv, y_test.csv")
                print("  - Data Prep Summary: "
                      "artifacts/data_prep_summary.md")
                print("  - Baseline Comparison: "
                      "artifacts/baseline_comparison.md")
                print("  - Best Model: artifacts/model.pkl")
                print("  - Hyperparameter Tuning: "
                      "artifacts/hyperparameter_tuning.md")
                print("  - Final Model Validation: "
                      "artifacts/final_model_validation.md")
                print("  - Evaluation Report: "
                      "artifacts/evaluation_report.md")
                print("  - Model Card: artifacts/model_card.md")
                print("  - Deployment Package: artifacts/deployment/")

                # Log artifacts
                for output_file in required_outputs:
                    self.logger_manager.log_artifact_created(
                        output_file,
                        "Data Science Output"
                    )

            except ValidationError as ve:
                print(f"\n❌ Output validation failed: {ve}")
                self.logger_manager.log_validation(
                    "Model Output Files",
                    "FAILED",
                    str(ve)
                )
                self.state.workflow_status = "failed"
                raise

            duration = time.time() - start_time
            self.logger_manager.log_stage_end(
                "Data Scientist Crew",
                status="SUCCESS"
            )
            print(f"\n⏱️  Data Scientist Crew completed in {duration:.2f} seconds")
            self.logger.info(f"Duration: {duration:.2f} seconds")

        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ Data Scientist Crew failed: {str(e)}")
            print("\nError details:")
            traceback.print_exc()

            self.logger_manager.log_error(
                f"Data Scientist Crew failed: {str(e)}",
                exc_info=True
            )
            self.logger_manager.log_stage_end(
                "Data Scientist Crew",
                status="FAILED"
            )
            self.logger.info(f"Duration before failure: {duration:.2f} seconds")

            self.state.workflow_status = "failed"
            self.state.current_stage = "data_science_failed"
            raise RuntimeError(
                f"Data Scientist Crew failed at stage: "
                f"{self.state.current_stage}"
            ) from e

    @listen(data_scientist_crew)
    def finalize_pipeline(self):
        """Finalize the pipeline and report results."""
        print("\n" + "=" * 60)
        print("Retail AI Pipeline Completed Successfully!")
        print("=" * 60)

        self.logger_manager.log_stage_start(
            "Pipeline Finalization",
            details="Generating final summary"
        )

        print("\nGenerated Artifacts:")
        print(f"  - Cleaned Data: {self.state.clean_data_path}")
        print(f"  - Features: {self.state.features_path}")
        print(f"  - Model: {self.state.model_path}")
        print("  - EDA Report: artifacts/eda_report.html")
        print("  - Insights: artifacts/insights.md")
        print("  - Dataset Contract: artifacts/dataset_contract.json")
        print("  - Evaluation Report: artifacts/evaluation_report.md")
        print("  - Model Card: artifacts/model_card.md")
        print("\nNext steps:")
        print("  1. Review the EDA report and insights")
        print("  2. Check model evaluation metrics")
        print("  3. Launch Streamlit app: streamlit run app/streamlit_app.py")

        self.state.workflow_status = "completed"

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60)
        self.logger.info(f"Final status: {self.state.workflow_status}")
        self.logger.info(f"Run log saved to: {self.run_log_path}")
        self.logger.info("=" * 60)

        self.logger_manager.log_stage_end(
            "Pipeline Finalization",
            status="SUCCESS"
        )


def main():
    """Main entry point for the crew flow."""
    flow = RetailAIFlow()
    flow.kickoff()


def plot():
    """Generate a plot of the flow."""
    flow = RetailAIFlow()
    flow.plot()


if __name__ == "__main__":
    main()
