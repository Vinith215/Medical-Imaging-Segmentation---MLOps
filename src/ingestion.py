import os
import boto3
pydicom = None
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.global_config import GlobalConfig

# dicom2nifti is imported lazily inside convert_dicom_to_nifti to avoid unresolved import errors
dicom2nifti = None

def ensure_pydicom():
    """
    Lazily import pydicom at runtime to avoid static import resolution errors in editors;
    raises a clear RuntimeError if pydicom is not installed.
    """
    global pydicom
    if pydicom is None:
        try:
            import importlib
            pydicom = importlib.import_module("pydicom")
        except Exception:
            raise RuntimeError("Missing dependency 'pydicom'. Install it with: pip install pydicom")

class DataIngestion:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=GlobalConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=GlobalConfig.AWS_SECRET_KEY,
            region_name=GlobalConfig.S3_REGION
        )
        # Ensure local directories exist
        GlobalConfig.create_directories()

    def sync_from_s3(self, s3_prefix="raw-data/"):
        """Downloads raw DICOM folders from S3 bucket."""
        print(f"Syncing from S3 bucket: {GlobalConfig.S3_BUCKET_NAME}...")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=GlobalConfig.S3_BUCKET_NAME, Prefix=s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    local_path = GlobalConfig.ROOT_DIR / s3_key
                    
                    # Create directory structure locally
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not local_path.exists():
                        self.s3_client.download_file(GlobalConfig.S3_BUCKET_NAME, s3_key, str(local_path))
        print("✅ S3 Sync Complete.")

    def validate_dicom_series(self, dicom_dir):
        """Checks if a directory contains a valid, readable DICOM series."""
        ensure_pydicom()
        files = list(Path(dicom_dir).glob("*.dcm"))
        if len(files) < 10:  # Minimum slice threshold for 3D
            return False
        try:
            sample = pydicom.dcmread(files[0])
            # Check for critical medical metadata
            required_tags = ['PatientID', 'Modality', 'SeriesInstanceUID']
            return all(tag in sample for tag in required_tags)
        except:
            return False

    def convert_dicom_to_nifti(self):
        """Converts validated DICOM folders to NIfTI and updates manifest."""
        ensure_pydicom()
        manifest_data = []
        raw_folders = [f for f in GlobalConfig.RAW_DATA_DIR.iterdir() if f.is_dir()]

        print(f"Converting {len(raw_folders)} patient folders...")
        for patient_dir in tqdm(raw_folders):
            if self.validate_dicom_series(patient_dir):
                output_nifti = GlobalConfig.PROCESSED_DATA_DIR / f"{patient_dir.name}.nii.gz"
                
                try:
                    # Lazy import to avoid top-level unresolved import; provide a clear error if missing
                    if dicom2nifti is None:
                        try:
                            import importlib
                            dicom2nifti = importlib.import_module("dicom2nifti")
                            globals()["dicom2nifti"] = dicom2nifti
                        except Exception:
                            raise RuntimeError(
                                "Missing dependency 'dicom2nifti'. Install it with: pip install dicom2nifti"
                            )

                    # Perform 3D conversion
                    dicom2nifti.dicom_series_to_nifti(
                        str(patient_dir),
                        str(output_nifti),
                        reorient_nifti=True
                    )

                    # Extract metadata for the Audit Trail
                    ds = pydicom.dcmread(next(patient_dir.glob("*.dcm")))
                    manifest_data.append({
                        "patient_id": ds.PatientID,
                        "modality": ds.Modality,
                        "raw_path": str(patient_dir),
                        "processed_path": str(output_nifti),
                        "series_uid": ds.SeriesInstanceUID
                    })
                except Exception as e:
                    print(f"❌ Failed to convert {patient_dir.name}: {e}")

        # Save the manifest to data/metadata/
        df = pd.DataFrame(manifest_data)
        df.to_csv(GlobalConfig.METADATA_DIR / "dataset_manifest.csv", index=False)
        print(f"✅ Ingestion successful. Manifest updated at {GlobalConfig.METADATA_DIR}")

if __name__ == "__main__":
    ingestor = DataIngestion()
    # ingestor.sync_from_s3() # Uncomment to pull from AWS
    ingestor.convert_dicom_to_nifti()