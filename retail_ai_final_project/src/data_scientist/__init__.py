"""
Data Scientist Module
Handles feature engineering, model training, and evaluation.
"""
from .feature_engineering import engineer_features, save_features
from .train_models import train_model, save_model
from .evaluate_models import evaluate_model, generate_evaluation_report, generate_model_card

__all__ = [
    'engineer_features',
    'save_features',
    'train_model',
    'save_model',
    'evaluate_model',
    'generate_evaluation_report',
    'generate_model_card',
]
