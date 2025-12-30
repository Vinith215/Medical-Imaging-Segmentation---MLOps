import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (AWS keys, MLflow URIs)
load_dotenv()

class GlobalConfig:
    # --- Project Root & Directory Structure ---
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # Data Directories
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    METADATA_DIR = DATA_DIR / "metadata"
    
    # Model & Artifacts
    MODEL_DIR = ROOT_DIR / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    PRODUCTION_MODEL_DIR = MODEL_DIR / "production"
    REPORTS_DIR = ROOT_DIR / "reports"
    
    # --- AWS Settings ---
    # These are pulled from your .env file for security
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "med-segmentation-data-lake")
    S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # --- MLflow / Tracking Settings ---
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = "Brain_Tumor_Segmentation_3D"
    
    # --- Medical Imaging Parameters ---
    # Unified spatial settings for the whole pipeline
    TARGET_SPACING = (1.0, 1.0, 1.0)  # Standardize to 1mm isotropic voxels
    IMAGE_SIZE = (128, 128, 128)      # Crop/Pad size for 3D volumes
    
    # Windowing (Hounsfield Units) - Example for CT (Soft Tissue)
    # If using MRI, these might be ignored or used for percentile clipping
    HU_MIN = -200
    HU_MAX = 200
    
    # --- System & Hardware ---
    DEVICE = "cuda" if os.getenv("USE_GPU", "True") == "True" else "cpu"
    NUM_WORKERS = 4  # For DataLoader parallelization
    SEED = 42        # For reproducibility

    @classmethod
    def create_directories(cls):
        """Ensures the project folder structure exists on the local machine."""
        dirs = [
            cls.RAW_DATA_DIR, 
            cls.PROCESSED_DATA_DIR, 
            cls.METADATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.PRODUCTION_MODEL_DIR,
            cls.REPORTS_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"Directory verified: {d}")

# Optional: Run this file directly to initialize the project folders
if __name__ == "__main__":
    GlobalConfig.create_directories()