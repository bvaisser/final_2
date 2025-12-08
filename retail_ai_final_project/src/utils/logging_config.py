"""
Structured Logging Configuration for Retail AI Pipeline
Provides timestamped, formatted logging with file rotation and console output.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class PipelineLogger:
    """
    Centralized logging configuration for the ML pipeline.
    Provides structured logging with timestamps, file rotation, and console output.
    """

    def __init__(
        self,
        name: str = "retail_ai_pipeline",
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        max_bytes: int = 10_485_760,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize pipeline logger.

        Args:
            name: Logger name
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of backup log files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return the logger instance."""
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | '
                '%(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger

    def log_stage_start(self, stage_name: str, details: Optional[str] = None):
        """Log the start of a pipeline stage."""
        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info(f"STAGE START: {stage_name}")
        if details:
            self.logger.info(f"Details: {details}")
        self.logger.info(separator)

    def log_stage_end(self, stage_name: str, status: str = "SUCCESS"):
        """Log the end of a pipeline stage."""
        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info(f"STAGE END: {stage_name} | Status: {status}")
        self.logger.info(separator)

    def log_crew_start(self, crew_name: str, agent_count: int, task_count: int):
        """Log the start of a crew execution."""
        self.logger.info(f"🚀 Starting {crew_name}")
        self.logger.info(f"   Agents: {agent_count}")
        self.logger.info(f"   Tasks: {task_count}")

    def log_crew_end(self, crew_name: str, duration: Optional[float] = None):
        """Log the end of a crew execution."""
        msg = f"✅ Completed {crew_name}"
        if duration:
            msg += f" (Duration: {duration:.2f}s)"
        self.logger.info(msg)

    def log_validation(
        self,
        validation_type: str,
        status: str,
        details: Optional[str] = None
    ):
        """Log validation results."""
        emoji = "✅" if status == "PASSED" else "❌"
        self.logger.info(f"{emoji} Validation: {validation_type} | {status}")
        if details:
            self.logger.info(f"   Details: {details}")

    def log_artifact_created(self, artifact_path: str, artifact_type: str):
        """Log artifact creation."""
        self.logger.info(f"📦 Created {artifact_type}: {artifact_path}")

    def log_error(self, error_msg: str, exc_info: bool = True):
        """Log an error with optional exception info."""
        self.logger.error(f"❌ ERROR: {error_msg}", exc_info=exc_info)

    def log_warning(self, warning_msg: str):
        """Log a warning."""
        self.logger.warning(f"⚠️  WARNING: {warning_msg}")

    def log_metric(self, metric_name: str, value: float, context: str = ""):
        """Log a metric value."""
        msg = f"📊 Metric: {metric_name} = {value}"
        if context:
            msg += f" ({context})"
        self.logger.info(msg)

    def create_run_log(self, run_name: Optional[str] = None) -> Path:
        """
        Create a timestamped log file for a specific run.

        Args:
            run_name: Optional name for the run

        Returns:
            Path to the run-specific log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            log_filename = f"run_{run_name}_{timestamp}.log"
        else:
            log_filename = f"run_{timestamp}.log"

        run_log_path = self.log_dir / log_filename

        # Create run-specific file handler
        run_handler = logging.FileHandler(
            filename=run_log_path,
            encoding='utf-8'
        )
        run_handler.setLevel(self.log_level)

        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | '
                '%(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        run_handler.setFormatter(detailed_formatter)

        self.logger.addHandler(run_handler)
        self.logger.info(f"📝 Run log created: {run_log_path}")

        return run_log_path


def setup_pipeline_logging(
    log_level: str = "INFO",
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Quick setup function for pipeline logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    pipeline_logger = PipelineLogger(
        log_level=level_map.get(log_level.upper(), logging.INFO),
        log_dir=log_dir
    )

    return pipeline_logger.get_logger()


# Example usage
if __name__ == "__main__":
    # Initialize logger
    pipeline_logger = PipelineLogger()

    # Log pipeline stages
    pipeline_logger.log_stage_start(
        "Data Ingestion",
        details="Loading dataset from Kaggle"
    )

    pipeline_logger.log_crew_start(
        crew_name="Data Analyst Crew",
        agent_count=3,
        task_count=10
    )

    pipeline_logger.log_validation(
        validation_type="Schema Validation",
        status="PASSED",
        details="All columns match expected schema"
    )

    pipeline_logger.log_artifact_created(
        artifact_path="data/interim/clean_data.csv",
        artifact_type="Clean Dataset"
    )

    pipeline_logger.log_metric(
        metric_name="Accuracy",
        value=0.95,
        context="Random Forest model"
    )

    pipeline_logger.log_crew_end(
        crew_name="Data Analyst Crew",
        duration=125.5
    )

    pipeline_logger.log_stage_end("Data Ingestion", status="SUCCESS")

    print(f"\nLogs written to: {pipeline_logger.log_dir}")
