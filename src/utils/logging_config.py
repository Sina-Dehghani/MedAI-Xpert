import os
import yaml
import logging.config


def setup_logging():
    """Sets up logging configuration from a YAML file."""
    log_config_path = os.path.join(
        os.path.dirname(__file__), "../../config/logging.yaml"
    )
    if not os.path.exists("logs"):
        os.makedirs("logs")  # Ensure logs directory exists

    with open(log_config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)
    logging.getLogger(__name__).info("Logging configured successfully.")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("src.utils.logging_config")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
