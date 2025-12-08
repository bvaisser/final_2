"""
Reproducibility Configuration for Retail AI Pipeline
Ensures deterministic behavior across all stages of the ML pipeline.
"""
import os
import random
import numpy as np
from typing import Optional
import logging


class ReproducibilityConfig:
    """
    Manages reproducibility settings for the ML pipeline.
    Sets random seeds for Python, NumPy, and scikit-learn.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize reproducibility configuration.

        Args:
            seed: Random seed value (default: 42)
        """
        self.seed = seed
        self.logger = logging.getLogger("retail_ai_pipeline")

    def set_global_seeds(self) -> None:
        """
        Set random seeds for all libraries to ensure reproducibility.

        This sets seeds for:
        - Python's built-in random module
        - NumPy
        - Environment variables for hash seed
        """
        # Set Python random seed
        random.seed(self.seed)
        self.logger.info(f"Set Python random seed: {self.seed}")

        # Set NumPy random seed
        np.random.seed(self.seed)
        self.logger.info(f"Set NumPy random seed: {self.seed}")

        # Set hash seed for Python
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        self.logger.info(f"Set PYTHONHASHSEED: {self.seed}")

        # Try to set scikit-learn random state if available
        try:
            from sklearn.utils import check_random_state
            check_random_state(self.seed)
            self.logger.info(f"Set scikit-learn random state: {self.seed}")
        except ImportError:
            self.logger.warning(
                "scikit-learn not available, skipping sklearn seed"
            )

    def get_random_state(self) -> int:
        """
        Get the random state/seed value.

        Returns:
            The configured seed value
        """
        return self.seed

    def log_configuration(self) -> None:
        """Log the current reproducibility configuration."""
        self.logger.info("=" * 60)
        self.logger.info("REPRODUCIBILITY CONFIGURATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Global Random Seed: {self.seed}")
        self.logger.info(f"Python Random: {random.getstate()[0]}")
        self.logger.info(f"NumPy Random: seeded with {self.seed}")
        self.logger.info(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
        self.logger.info("=" * 60)


def setup_reproducibility(
    seed: int = 42,
    log_config: bool = True
) -> ReproducibilityConfig:
    """
    Quick setup function for reproducibility.

    Args:
        seed: Random seed value (default: 42)
        log_config: Whether to log the configuration (default: True)

    Returns:
        Configured ReproducibilityConfig instance

    Example:
        >>> config = setup_reproducibility(seed=42)
        >>> random_state = config.get_random_state()
    """
    config = ReproducibilityConfig(seed=seed)
    config.set_global_seeds()

    if log_config:
        config.log_configuration()

    return config


# Default seeds for different stages
DEFAULT_SEEDS = {
    "global": 42,
    "data_split": 42,
    "feature_engineering": 42,
    "model_training": 42,
    "cross_validation": 42,
    "hyperparameter_tuning": 42
}


def get_seed_for_stage(stage: str) -> int:
    """
    Get the recommended seed for a specific pipeline stage.

    Args:
        stage: Pipeline stage name

    Returns:
        Seed value for that stage
    """
    return DEFAULT_SEEDS.get(stage, DEFAULT_SEEDS["global"])


# Best practices for reproducibility
REPRODUCIBILITY_CHECKLIST = """
# Reproducibility Checklist for ML Pipeline

✅ Random Seeds:
   - Set global random seed (Python, NumPy)
   - Set PYTHONHASHSEED environment variable
   - Pass random_state to all sklearn functions
   - Use consistent seeds across pipeline stages

✅ Data Splitting:
   - Use train_test_split with fixed random_state
   - Document split ratios
   - Save split indices or use stratified sampling

✅ Feature Engineering:
   - Document all transformations
   - Save fitted transformers (scalers, encoders)
   - Use deterministic operations where possible

✅ Model Training:
   - Set random_state in model constructors
   - Set random_state in cross-validation
   - Set random_state in hyperparameter search
   - Document all hyperparameters

✅ Evaluation:
   - Use fixed random_state for resampling
   - Save evaluation metrics with timestamps
   - Document evaluation methodology

✅ Environment:
   - Pin all package versions in requirements.txt
   - Document Python version
   - Use virtual environment
   - Save environment snapshot

✅ Logging:
   - Log all random seeds used
   - Log timestamps for all operations
   - Save configuration files
   - Version control all code

✅ Artifacts:
   - Save all models with versioning
   - Save preprocessors and transformers
   - Save feature names and metadata
   - Keep audit trail of changes
"""


def print_reproducibility_checklist():
    """Print the reproducibility best practices checklist."""
    print(REPRODUCIBILITY_CHECKLIST)


# Example usage and documentation
if __name__ == "__main__":
    print("=" * 60)
    print("Reproducibility Configuration Demo")
    print("=" * 60)

    # Setup reproducibility
    config = setup_reproducibility(seed=42, log_config=True)

    # Get seed for specific stages
    print("\nRecommended Seeds by Stage:")
    for stage, seed in DEFAULT_SEEDS.items():
        print(f"  {stage}: {seed}")

    # Demonstrate deterministic behavior
    print("\n" + "=" * 60)
    print("Demonstrating Deterministic Behavior")
    print("=" * 60)

    # Reset seeds
    config.set_global_seeds()

    # Generate random numbers (should be same each run)
    print(f"\nPython random: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.randint(0, 100)}")
    print(f"NumPy array: {np.random.rand(3)}")

    # Print checklist
    print("\n")
    print_reproducibility_checklist()
