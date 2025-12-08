# Retail AI Final Project

A comprehensive machine learning pipeline for retail analytics, powered by CrewAI agents for automated data analysis, feature engineering, and model training.

## Project Overview

This project implements an end-to-end ML pipeline using CrewAI's multi-agent framework to automate the complete workflow from data ingestion to model deployment.

### Key Features

- **Automated ML Pipeline**: Three specialized crews handling data analysis, feature engineering, and model evaluation
- **CrewAI Integration**: Intelligent agents coordinating complex ML workflows
- **Interactive UI**: Streamlit application for easy interaction
- **REST API**: Flask API for programmatic access
- **Comprehensive Reporting**: Automated generation of EDA reports, model cards, and evaluation metrics

### Quick Results Summary

- **Model Performance**: R² = 0.9989 (99.89% accuracy) - Excellent model fit
- **Dataset**: 3,636 records processed, 19 engineered features
- **Best Predictors**: Time-based features (hour of day, weekday, month)
- **Test Performance**: RMSE = 0.0760, MAE = 0.0426 on 728 test samples

## Project Structure

```
retail_ai_final_project/
├── data/
│   ├── raw/                    # Raw data from Kaggle
│   ├── interim/                # Cleaned data
│   └── processed/              # Feature-engineered data
│
├── artifacts/                  # Generated outputs
│   ├── eda_report.html
│   ├── insights.md
│   ├── dataset_contract.json
│   ├── model.pkl
│   ├── evaluation_report.md
│   └── model_card.md
│
├── app/
│   ├── streamlit_app.py       # Interactive web UI
│   └── flask_app.py           # REST API
│
├── crew/
│   ├── agents.yaml            # Agent definitions
│   ├── tasks.yaml             # Task definitions
│   └── crew_flow.py           # Main orchestration
│
├── src/
│   ├── data_analyst/          # Data ingestion, cleaning, EDA
│   ├── data_scientist/        # Feature engineering, modeling
│   └── utils/                 # Shared utilities
│
└── notebooks/                 # Jupyter notebooks
```

## Installation

### Prerequisites

- Python 3.10 - 3.13
- pip or uv package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd retail_ai_final_project
   ```

2. **Install dependencies:**

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

   Or using uv (recommended):
   ```bash
   pip install uv
   uv sync
   ```

3. **Configure environment variables:**

   Edit `.env` file and add your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Place your training data:**

   Put your `Coffe_sales.csv` file in the `data/raw/` directory:
   ```bash
   cp /path/to/your/Coffe_sales.csv data/raw/Coffe_sales.csv
   ```

## Usage

### Option 1: Run via Streamlit UI (Recommended)

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

Then navigate to `http://localhost:8501` in your browser.

The Streamlit app provides:
- Data upload interface
- Pipeline execution controls
- Results visualization
- Model predictions interface

### Option 2: Run via Command Line

Execute the complete pipeline:

```bash
python crew/crew_flow.py
```

Or run individual modules:

```bash
# Data ingestion and cleaning
python src/data_analyst/ingest_and_clean.py

# Exploratory data analysis
python src/data_analyst/eda.py

# Dataset contract creation
python src/data_analyst/build_contract.py

# Feature engineering
python src/data_scientist/feature_engineering.py

# Model training
python src/data_scientist/train_models.py

# Model evaluation
python src/data_scientist/evaluate_models.py
```

### Option 3: Run via Flask API

Start the API server:

```bash
python app/flask_app.py
```

API endpoints available at `http://localhost:5000`:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `POST /predict/batch` - Batch predictions from CSV
- `POST /pipeline/run` - Trigger pipeline execution
- `GET /pipeline/status` - Get pipeline status
- `GET /model/info` - Get model information
- `GET /artifacts/<name>` - Retrieve specific artifact

Example API call:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"feature1": 1, "feature2": 2}]}'
```

## Pipeline Workflow

### Stage 1: Data Analyst Crew

**Agent:** Senior Data Analyst

**Tasks:**
1. **Ingest and Clean**: Load raw data, handle missing values, remove duplicates
2. **EDA**: Generate statistical summaries and visualizations
3. **Contract Creation**: Define formal data schema and constraints

**Outputs:**
- `data/interim/clean_data.csv`
- `artifacts/eda_report.html`
- `artifacts/insights.md`
- `artifacts/dataset_contract.json`

### Stage 2: Data Scientist Crew

**Agent:** Senior Data Scientist

**Tasks:**
1. **Feature Engineering**: Create new features, encode categoricals, scale numerics
2. **Model Training**: Train multiple models with hyperparameter tuning

**Outputs:**
- `data/processed/features.csv`
- `artifacts/model.pkl`

### Stage 3: Model Evaluator Crew

**Agent:** ML Model Evaluator

**Tasks:**
1. **Model Evaluation**: Calculate comprehensive metrics
2. **Report Generation**: Create evaluation reports and model cards

**Outputs:**
- `artifacts/evaluation_report.md`
- `artifacts/model_card.md`
- `artifacts/confusion_matrix.png`
- `artifacts/roc_curve.png`
- `artifacts/feature_importance.png`

## Results

### Model Performance

The trained Random Forest model achieved excellent performance on the retail sales prediction task:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.9989 | Excellent fit (99.89% variance explained) |
| **RMSE** | 0.0760 | Low prediction error |
| **MAE** | 0.0426 | Mean absolute error is minimal |
| **MSE** | 0.0058 | Very low mean squared error |

### Key Findings

1. **Feature Importance**: The most critical features for prediction are:
   - `hour_of_day` (30.55% importance) - Time of day is the strongest predictor
   - `hour_of_day_binned_encoded` (25.47% importance) - Binned time features are highly informative
   - `hour_of_day_squared` (22.20% importance) - Non-linear time patterns matter
   - `Weekdaysort` and `Monthsort` - Day of week and month show significant impact

2. **Data Quality**: 
   - Dataset contains 3,636 records with 12 original features
   - No missing values detected
   - Successfully engineered 19 features for modeling

3. **Model Characteristics**:
   - Model Type: Random Forest Regressor
   - Test Set Size: 728 samples (20% of data)
   - Number of Features: 19 engineered features

### Visualizations

The pipeline generates several visualization artifacts:
- **Residual Plot** (`artifacts/residual_plot.png`) - Shows prediction errors distribution
- **Actual vs Predicted** (`artifacts/actual_vs_predicted.png`) - Visualizes model accuracy
- **Feature Importance** (`artifacts/feature_importance.png`) - Ranks feature contributions
- **Correlation Heatmap** (`artifacts/figures/correlation_heatmap.png`) - Shows feature relationships
- **Time-based Distributions** - Hour, weekday, and month patterns

### Business Insights

1. **Temporal Patterns**: Time-based features (hour, day, month) are the strongest predictors, indicating strong temporal patterns in retail sales
2. **Model Reliability**: With R² of 0.9989, the model provides highly reliable predictions
3. **Feature Engineering Impact**: Engineered features (squared terms, interactions) significantly improved model performance

### Generated Artifacts

All results are saved in the `artifacts/` directory:
- `evaluation_report.md` - Comprehensive evaluation metrics
- `model_card.md` - Model documentation and metadata
- `insights.md` - Data insights and recommendations
- `eda_report.html` - Interactive exploratory data analysis
- `dataset_contract.json` - Data schema and validation rules
- Various visualization PNG files

For detailed results, see:
- Full evaluation: `artifacts/evaluation_report.md`
- Model documentation: `artifacts/model_card.md`
- Data insights: `artifacts/insights.md`

### Viewing Results

After running the pipeline, view results:
- Open `artifacts/eda_report.html` in a browser for interactive EDA
- Check `artifacts/evaluation_report.md` for performance metrics
- Review `artifacts/model_card.md` for model documentation
- View PNG files in `artifacts/` and `artifacts/figures/` for visualizations

### Generate Comprehensive PDF Report

Create a professional PDF report with all charts, visualizations, and analysis results:

```bash
python src/utils/generate_pdf_report.py
```

Or specify a custom output path:

```bash
python src/utils/generate_pdf_report.py --output artifacts/my_report.pdf
```

The PDF includes:
- Executive summary with key metrics
- Model performance metrics table
- Data insights and analysis
- All visualizations (residual plots, feature importance, distributions)
- Model evaluation report
- Model card documentation
- Recommendations and next steps

The generated PDF will be saved in the `artifacts/` directory with a timestamp.

## CrewAI Agents

### Data Analyst Agent
- **Role**: Senior Data Analyst
- **Goal**: Produce clean datasets and comprehensive insights
- **Skills**: Data cleaning, EDA, contract creation

### Data Scientist Agent
- **Role**: Senior Data Scientist
- **Goal**: Build robust ML models for retail predictions
- **Skills**: Feature engineering, model selection, evaluation

### Model Evaluator Agent
- **Role**: ML Model Evaluator
- **Goal**: Generate comprehensive evaluation reports
- **Skills**: Metrics calculation, visualization, documentation

## Customization

### Adding New Features

1. **Custom Data Processing**: Modify `src/data_analyst/ingest_and_clean.py`
2. **Feature Engineering**: Extend `src/data_scientist/feature_engineering.py`
3. **Model Types**: Add models in `src/data_scientist/train_models.py`

### Configuring Agents

Edit `crew/agents.yaml` to customize agent behavior:
```yaml
data_analyst:
  role: Your custom role
  goal: Your custom goal
  backstory: Your custom backstory
```

### Configuring Tasks

Edit `crew/tasks.yaml` to modify task descriptions and expected outputs.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ app/ crew/

# Lint code
flake8 src/ app/ crew/

# Type checking
mypy src/ app/ crew/
```

### Jupyter Notebooks

Explore data interactively:

```bash
jupyter notebook notebooks/eda_playground.ipynb
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**: Run the pipeline before making predictions
   ```bash
   python crew/crew_flow.py
   ```

3. **API Key Issues**: Check `.env` file has correct `OPENAI_API_KEY`

4. **Memory Issues**: For large datasets, consider processing in chunks

## Performance Optimization

- **Parallel Processing**: Use `n_jobs=-1` in sklearn models
- **Data Sampling**: For experimentation, sample large datasets
- **Feature Selection**: Remove low-importance features
- **Model Caching**: Reuse trained models when possible

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

Build and run:
```bash
docker build -t retail-ai .
docker run -p 8501:8501 retail-ai
```

### Cloud Deployment

- **Streamlit Cloud**: Deploy directly from GitHub
- **AWS/GCP/Azure**: Use container services (ECS, Cloud Run, ACI)
- **Heroku**: Use Procfile for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [CrewAI](https://crewai.com)
- Powered by [OpenAI](https://openai.com)
- UI built with [Streamlit](https://streamlit.io)

## Support

For issues and questions:
- Open an issue on GitHub
- Check documentation at `/docs`
- Review example notebooks in `/notebooks`

## Roadmap

- [ ] Add more model types (XGBoost, LightGBM)
- [ ] Implement A/B testing framework
- [ ] Add real-time prediction API
- [ ] Create Docker Compose setup
- [ ] Add automated testing pipeline
- [ ] Implement model versioning
- [ ] Add monitoring and alerting

## Contact

Project maintainer: Your Name bvaisser@gmail.com

---

**Built with CrewAI** | **Version 1.0** | **Last Updated: 2024**
