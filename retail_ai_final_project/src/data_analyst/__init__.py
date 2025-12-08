"""
Data Analyst Module
Handles data ingestion, cleaning, EDA, and contract creation.
"""
from .ingest_and_clean import load_data, clean_data, save_clean_data
from .eda import perform_eda, generate_eda_report, generate_insights
from .build_contract import create_dataset_contract, validate_contract

__all__ = [
    'load_data',
    'clean_data',
    'save_clean_data',
    'perform_eda',
    'generate_eda_report',
    'generate_insights',
    'create_dataset_contract',
    'validate_contract',
]
