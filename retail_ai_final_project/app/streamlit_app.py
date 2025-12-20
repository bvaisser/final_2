"""
Streamlit Application
User-friendly interface for the Retail AI project.
"""
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Set up project root for imports (DO NOT change working directory)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger
from crew.crew_flow import RetailAIFlow

logger = get_logger(__name__)


def generate_prediction_insights(predictions: np.ndarray, df: pd.DataFrame) -> str:
    """
    Generate natural language insights from predictions.

    Args:
        predictions: Array of predictions
        df: Original dataframe with features

    Returns:
        String containing natural language insights
    """
    insights = []

    # Basic statistics
    mean_pred = predictions.mean()
    median_pred = np.median(predictions)
    std_pred = predictions.std()
    min_pred = predictions.min()
    max_pred = predictions.max()

    # Overall summary
    insights.append(f"## 📊 Prediction Analysis Summary\n")
    insights.append(f"Based on the analysis of **{len(predictions)} predictions**, here are the key insights:\n")

    # Central tendency
    insights.append(f"### Central Tendency")
    insights.append(f"- The **average prediction** is **{mean_pred:.2f}**")
    insights.append(f"- The **median prediction** is **{median_pred:.2f}**")

    if abs(mean_pred - median_pred) / mean_pred > 0.1:
        insights.append(f"- ⚠️ There's a notable difference between mean and median, suggesting the data may be **skewed**")
    else:
        insights.append(f"- ✓ Mean and median are close, indicating a **relatively symmetric distribution**")

    # Variability
    insights.append(f"\n### Variability")
    insights.append(f"- The **standard deviation** is **{std_pred:.2f}**, indicating {'high' if std_pred/mean_pred > 0.3 else 'moderate' if std_pred/mean_pred > 0.15 else 'low'} variability")
    insights.append(f"- Predictions range from **{min_pred:.2f}** to **{max_pred:.2f}** (range: **{max_pred - min_pred:.2f}**)")

    # Distribution analysis
    insights.append(f"\n### Distribution Insights")

    # Quartiles
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    insights.append(f"- **25% of predictions** are below **{q1:.2f}**")
    insights.append(f"- **75% of predictions** are below **{q3:.2f}**")
    insights.append(f"- The **interquartile range (IQR)** is **{q3 - q1:.2f}**")

    # Outlier detection
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = ((predictions < lower_fence) | (predictions > upper_fence)).sum()

    if outliers > 0:
        outlier_pct = (outliers / len(predictions)) * 100
        insights.append(f"- ⚠️ **{outliers} potential outliers** detected ({outlier_pct:.1f}% of predictions)")
    else:
        insights.append(f"- ✓ **No significant outliers** detected in the predictions")

    # Prediction categories (if applicable)
    insights.append(f"\n### Prediction Categories")

    if mean_pred > 0:
        # Create meaningful categories based on distribution
        low_threshold = q1
        high_threshold = q3

        low_count = (predictions < low_threshold).sum()
        medium_count = ((predictions >= low_threshold) & (predictions < high_threshold)).sum()
        high_count = (predictions >= high_threshold).sum()

        insights.append(f"- **Low predictions** (< {low_threshold:.2f}): **{low_count}** ({low_count/len(predictions)*100:.1f}%)")
        insights.append(f"- **Medium predictions** ({low_threshold:.2f} - {high_threshold:.2f}): **{medium_count}** ({medium_count/len(predictions)*100:.1f}%)")
        insights.append(f"- **High predictions** (≥ {high_threshold:.2f}): **{high_count}** ({high_count/len(predictions)*100:.1f}%)")

    # Top predictions
    insights.append(f"\n### Notable Predictions")
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    insights.append(f"- **Top 5 highest predictions**: {', '.join([f'{predictions[i]:.2f}' for i in top_5_idx])}")

    bottom_5_idx = np.argsort(predictions)[:5]
    insights.append(f"- **Top 5 lowest predictions**: {', '.join([f'{predictions[i]:.2f}' for i in bottom_5_idx])}")

    # Recommendations
    insights.append(f"\n### 💡 Recommendations")

    if std_pred/mean_pred > 0.3:
        insights.append(f"- High variability detected. Consider investigating the factors driving extreme predictions")

    if outliers > len(predictions) * 0.05:
        insights.append(f"- Multiple outliers detected. Review these cases for data quality or special circumstances")

    insights.append(f"- Monitor predictions over time to detect any drift in model behavior")
    insights.append(f"- Consider collecting feedback on prediction accuracy for continuous improvement")

    return "\n".join(insights)


# Page configuration
st.set_page_config(
    page_title="Retail AI Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application."""

    # Title and description
    st.title("🛍️ Retail AI Platform")
    st.markdown("""
    Welcome to the Retail AI Platform! This application provides an end-to-end
    machine learning pipeline for retail analytics and predictions.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Data Upload", "Run Pipeline", "View Results", "Model Predictions"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Run Pipeline":
        show_pipeline_page()
    elif page == "View Results":
        show_results_page()
    elif page == "Model Predictions":
        show_predictions_page()


def show_home_page():
    """Display home page with project overview."""
    st.header("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Pipeline Stages",
            value="3",
            help="Data Analysis, Model Training, Evaluation"
        )

    with col2:
        st.metric(
            label="Agents",
            value="3",
            help="Data Analyst, Data Scientist, Model Evaluator"
        )

    with col3:
        st.metric(
            label="Artifacts",
            value="8",
            help="Reports, models, and evaluation results"
        )

    st.markdown("---")

    st.subheader("🔄 Pipeline Workflow")
    st.markdown("""
    1. **Data Analysis Crew**
       - Ingest and clean raw data
       - Perform exploratory data analysis
       - Create dataset contract

    2. **Data Science Crew**
       - Engineer features
       - Train machine learning models
       - Optimize hyperparameters

    3. **Evaluation Crew**
       - Evaluate model performance
       - Generate comprehensive reports
       - Create model card
    """)

    st.markdown("---")

    st.subheader("📂 Project Structure")
    st.code("""
    retail_ai_final_project/
    ├── data/           # Raw, interim, and processed data
    ├── artifacts/      # Models, reports, and outputs
    ├── crew/           # CrewAI agents and tasks
    ├── src/            # Source code modules
    ├── app/            # Streamlit and Flask apps
    └── notebooks/      # Jupyter notebooks
    """, language="text")


def show_data_upload_page():
    """Display data upload page."""
    st.header("📤 Data Upload")

    st.markdown("""
    Upload your training data (CSV format) to begin the machine learning pipeline.
    The data will be saved to `data/raw/Coffe_sales.csv`.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read and display data preview
        df = pd.read_csv(uploaded_file)

        st.success(f"File uploaded successfully! Shape: {df.shape}")

        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.subheader("Data Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])

        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicates", df.duplicated().sum())

        # Save button
        if st.button("Save to data/raw/Coffe_sales.csv"):
            save_path = PROJECT_ROOT / "data/raw/Coffe_sales.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            st.success(f"Data saved to {save_path}")
            logger.info(f"Data uploaded and saved: {df.shape}")


def show_pipeline_page():
    """Display pipeline execution page."""
    st.header("🚀 Run ML Pipeline")

    st.markdown("""
    Execute the complete machine learning pipeline using CrewAI agents.
    This will run data analysis, feature engineering, model training, and evaluation.
    """)

    # Check if data exists
    data_path = PROJECT_ROOT / "data/raw/Coffe_sales.csv"
    if not data_path.exists():
        st.error("❌ No training data found. Please upload data first.")
        return

    st.success("✅ Training data found")

    # Pipeline configuration
    st.subheader("Pipeline Configuration")

    col1, col2 = st.columns(2)

    with col1:
        verbose = st.checkbox("Verbose Output", value=True)

    with col2:
        auto_save = st.checkbox("Auto-save Artifacts", value=True)

    st.markdown("---")

    # Run pipeline button
    if st.button("▶️ Run Pipeline", type="primary"):
        with st.spinner("Running ML pipeline... This may take several minutes."):
            try:
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Stage 1: Data Analysis
                status_text.text("Stage 1/3: Data Analysis...")
                progress_bar.progress(33)

                # Stage 2: Model Training
                status_text.text("Stage 2/3: Model Training...")
                progress_bar.progress(66)

                # Stage 3: Evaluation
                status_text.text("Stage 3/3: Model Evaluation...")
                progress_bar.progress(100)

                # Run the actual flow
                flow = RetailAIFlow()
                result = flow.kickoff()

                status_text.text("Pipeline completed successfully!")
                st.success("✅ Pipeline execution completed!")

                st.balloons()

                logger.info("Pipeline executed successfully via Streamlit")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                logger.error(f"Pipeline execution failed: {str(e)}")


def show_results_page():
    """Display results and artifacts."""
    st.header("📊 View Results")

    # Tabs for different artifacts
    tab1, tab2, tab3, tab4 = st.tabs([
        "EDA Report",
        "Model Evaluation",
        "Model Card",
        "Artifacts"
    ])

    with tab1:
        st.subheader("Exploratory Data Analysis")
        eda_path = PROJECT_ROOT / "artifacts/eda_report.html"
        insights_path = PROJECT_ROOT / "artifacts/insights.md"

        if insights_path.exists():
            with open(insights_path, 'r') as f:
                st.markdown(f.read())

            # Display EDA visualizations
            st.markdown("---")
            st.subheader("Data Visualizations")

            figures_dir = PROJECT_ROOT / "artifacts/figures"
            if figures_dir.exists():
                eda_figures = list(figures_dir.glob("*.png"))

                if eda_figures:
                    # Display in 2-column layout
                    cols = st.columns(2)
                    for idx, fig_path in enumerate(sorted(eda_figures)):
                        with cols[idx % 2]:
                            st.image(str(fig_path), caption=fig_path.stem.replace('_', ' ').title(), use_container_width=True)
                else:
                    st.info("No EDA figures found.")
            else:
                st.info("EDA figures directory not found.")
        else:
            st.info("EDA insights not yet generated. Run the pipeline first.")

    with tab2:
        st.subheader("Model Evaluation Report")
        eval_path = PROJECT_ROOT / "artifacts/evaluation_report.md"

        if eval_path.exists():
            with open(eval_path, 'r') as f:
                st.markdown(f.read())

            st.markdown("---")
            st.subheader("Model Performance Visualizations")

            # Classification visualizations
            col1, col2 = st.columns(2)
            with col1:
                cm_path = PROJECT_ROOT / "artifacts/confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion Matrix")

            with col2:
                roc_path = PROJECT_ROOT / "artifacts/roc_curve.png"
                if roc_path.exists():
                    st.image(str(roc_path), caption="ROC Curve")

            # Regression visualizations
            col3, col4 = st.columns(2)
            with col3:
                residual_path = PROJECT_ROOT / "artifacts/residual_plot.png"
                if residual_path.exists():
                    st.image(str(residual_path), caption="Residual Plot")

            with col4:
                actual_pred_path = PROJECT_ROOT / "artifacts/actual_vs_predicted.png"
                if actual_pred_path.exists():
                    st.image(str(actual_pred_path), caption="Actual vs Predicted")

            # Feature importance
            st.markdown("---")
            st.subheader("Feature Importance")
            feature_imp_path = PROJECT_ROOT / "artifacts/feature_importance.png"
            if feature_imp_path.exists():
                st.image(str(feature_imp_path), caption="Top 20 Feature Importances", use_column_width=True)
            else:
                st.info("Feature importance plot not available for this model type.")

        else:
            st.info("Evaluation report not yet generated. Run the pipeline first.")

    with tab3:
        st.subheader("Model Card")
        card_path = PROJECT_ROOT / "artifacts/model_card.md"

        if card_path.exists():
            with open(card_path, 'r') as f:
                st.markdown(f.read())
        else:
            st.info("Model card not yet generated. Run the pipeline first.")

    with tab4:
        st.subheader("Generated Artifacts")

        artifacts_dir = PROJECT_ROOT / "artifacts"
        if artifacts_dir.exists():
            files = list(artifacts_dir.glob("*"))
            if files:
                st.write("Available artifacts:")
                for file in files:
                    st.write(f"- {file.name}")
            else:
                st.info("No artifacts generated yet.")
        else:
            st.info("Artifacts directory not found. Run the pipeline first.")


def show_predictions_page():
    """Display model predictions interface."""
    st.header("🔮 Model Predictions")

    st.markdown("""
    Use the trained model to make predictions on new data.
    """)

    # Check if model exists
    model_path = PROJECT_ROOT / "artifacts/model.pkl"
    if not model_path.exists():
        st.error("❌ No trained model found. Please run the pipeline first.")
        return

    st.success("✅ Trained model found")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Display model info
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", type(model).__name__)
    with col2:
        if hasattr(model, 'n_features_in_'):
            st.metric("Expected Features", model.n_features_in_)

    st.markdown("---")
    st.subheader("Upload Data for Predictions")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="predict")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])

        if st.button("Generate Predictions", type="primary"):
            try:
                with st.spinner("Generating predictions..."):
                    # Note: In production, you'd need to preprocess the data
                    # the same way as training data
                    predictions = model.predict(df)

                    # Add predictions to dataframe
                    df['Prediction'] = predictions

                    st.success("✅ Predictions generated successfully!")

                    # Generate and display natural language insights
                    st.markdown("---")
                    insights_text = generate_prediction_insights(predictions, df.drop(columns=['Prediction']))
                    st.markdown(insights_text)

                    # Show prediction statistics
                    st.markdown("---")
                    st.subheader("Prediction Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Mean", f"{predictions.mean():.2f}")
                    with col3:
                        st.metric("Min", f"{predictions.min():.2f}")
                    with col4:
                        st.metric("Max", f"{predictions.max():.2f}")

                    # Show predictions distribution
                    st.markdown("---")
                    st.subheader("Predictions Distribution")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Predicted Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predictions')
                    st.pyplot(fig)

                    # Show full results
                    st.markdown("---")
                    st.subheader("Full Predictions")
                    st.dataframe(df, use_container_width=True)

                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("Make sure your data has the same features as the training data.")


if __name__ == "__main__":
    main()
