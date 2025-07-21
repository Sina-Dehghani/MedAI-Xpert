import os
from dotenv import load_dotenv


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self):
        """Loads configuration from environment variables."""
        load_dotenv()  # Load .env file

        self.APP_NAME = os.getenv("APP_NAME", "MediPredict-CDSS")
        self.API_VERSION = os.getenv("API_VERSION", "v1")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Database Settings
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/medipredict_db"
        )
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        self.CASSANDRA_CONTACT_POINTS = os.getenv(
            "CASSANDRA_CONTACT_POINTS", "localhost"
        ).split(",")

        # Model Paths (example)
        self.IMAGE_MODEL_PATH = os.getenv(
            "IMAGE_MODEL_PATH", "models/image_detector.pth"
        )
        self.NLP_MODEL_PATH = os.getenv("NLP_MODEL_PATH", "models/nlp_classifier.pth")

        # MLOps
        self.MLFLOW_TRACKING_URI = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

    def __getattr__(self, name):
        """Allow direct attribute access to settings."""
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


# Instantiate settings for easy import
settings = Settings()

if __name__ == "__main__":
    print(f"App Name: {settings.APP_NAME}")
    print(f"Database URL: {settings.DATABASE_URL}")
