"""
Flask API Application
REST API for model predictions and pipeline management.
"""
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger
from crew.crew_flow import RetailAIFlow

logger = get_logger(__name__)

app = Flask(__name__)


# Load model at startup
MODEL_PATH = Path("artifacts/model.pkl")
model = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
else:
    logger.warning("No model found at startup")


@app.route('/')
def home():
    """API home endpoint."""
    return jsonify({
        "message": "Retail AI API",
        "version": "1.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST)",
            "/pipeline/run": "Run ML pipeline (POST)",
            "/pipeline/status": "Get pipeline status",
            "/model/info": "Get model information"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using the trained model.

    Expected JSON format:
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ]
    }
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train a model first."
            }), 503

        # Get data from request
        data = request.get_json()

        if 'data' not in data:
            return jsonify({
                "error": "Missing 'data' field in request"
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame(data['data'])

        # Make predictions
        predictions = model.predict(df)

        # Convert predictions to list
        predictions_list = predictions.tolist()

        logger.info(f"Generated {len(predictions_list)} predictions")

        return jsonify({
            "predictions": predictions_list,
            "count": len(predictions_list)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions from uploaded CSV file.

    Expects file upload with key 'file'.
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train a model first."
            }), 503

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "error": "Empty filename"
            }), 400

        # Read CSV
        df = pd.read_csv(file)

        # Make predictions
        predictions = model.predict(df)

        # Add predictions to dataframe
        df['prediction'] = predictions

        # Convert to dict
        result = df.to_dict(orient='records')

        logger.info(f"Generated batch predictions for {len(result)} records")

        return jsonify({
            "predictions": result,
            "count": len(result)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """
    Trigger the ML pipeline execution.
    """
    try:
        logger.info("Pipeline execution triggered via API")

        # Run pipeline in background (in production, use Celery or similar)
        flow = RetailAIFlow()
        result = flow.kickoff()

        return jsonify({
            "status": "completed",
            "message": "Pipeline executed successfully"
        })

    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@app.route('/pipeline/status')
def pipeline_status():
    """
    Get current pipeline status.
    """
    # Check if artifacts exist
    artifacts_dir = Path("artifacts")
    artifacts = []

    if artifacts_dir.exists():
        artifacts = [f.name for f in artifacts_dir.glob("*")]

    return jsonify({
        "artifacts_generated": len(artifacts),
        "artifacts": artifacts,
        "model_exists": MODEL_PATH.exists()
    })


@app.route('/model/info')
def model_info():
    """
    Get information about the trained model.
    """
    if model is None:
        return jsonify({
            "error": "No model loaded"
        }), 404

    # Get model information
    info = {
        "model_type": type(model).__name__,
        "features": None,
        "parameters": None
    }

    # Try to get feature names
    if hasattr(model, 'feature_names_in_'):
        info['features'] = model.feature_names_in_.tolist()

    # Try to get model parameters
    if hasattr(model, 'get_params'):
        info['parameters'] = model.get_params()

    return jsonify(info)


@app.route('/artifacts/<artifact_name>')
def get_artifact(artifact_name):
    """
    Retrieve a specific artifact.
    """
    artifact_path = Path("artifacts") / artifact_name

    if not artifact_path.exists():
        return jsonify({
            "error": f"Artifact '{artifact_name}' not found"
        }), 404

    # For markdown files, return content
    if artifact_path.suffix == '.md':
        with open(artifact_path, 'r') as f:
            content = f.read()
        return jsonify({
            "name": artifact_name,
            "content": content,
            "type": "markdown"
        })

    # For JSON files, return as JSON
    elif artifact_path.suffix == '.json':
        import json
        with open(artifact_path, 'r') as f:
            content = json.load(f)
        return jsonify(content)

    else:
        return jsonify({
            "error": "Unsupported artifact type"
        }), 400


if __name__ == '__main__':
    logger.info("Starting Flask API server")
    app.run(debug=True, host='0.0.0.0', port=5000)
