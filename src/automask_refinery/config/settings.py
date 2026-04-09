import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "AutoMask-Refinery"
    DEBUG: bool = True
    PORT: int = 5000
    HOST: str = "0.0.0.0"

    # Data Paths
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "demo")
    
    # Output Paths
    REVIEW_OUT: str = os.path.join(DATA_DIR, "review_failures")
    PASSED_OUT: str = os.path.join(DATA_DIR, "review_passed")
    SUMMARY_CSV: str = os.path.join(DATA_DIR, "review_details.csv")

    class Config:
        env_prefix = "AUTOMASK_"
        env_file = ".env"

settings = Settings()
