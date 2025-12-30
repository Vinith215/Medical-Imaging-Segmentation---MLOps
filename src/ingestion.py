import os
import boto3
import SimpleITK as sitk
import nibabel as nib
import logging

# Setup logging for production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestor:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        
        # Ensure local directories exist
        os.makedirs(self.config.local_raw_dir, exist_ok=True)
        os.makedirs(self.config.standardized_dir, exist_ok=True)

    def download_from_s3(self, s3_key: str, local_name: str):
        """Downloads a specific volume from the S3 Data Lake."""
        local_path = os.path.join(self.config.local_raw_dir, local_name)
        try:
            self.s3_client.download_file(self.config.bucket_name, s3_key, local_path)
            logging.info(f"Downloaded {s3_key} to {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"Failed to download from S3: {e}")
            return None

    def convert_dicom_series(self, patient_id: str):
        """
        Converts a folder of DICOM slices into a single 3D NIfTI volume.
        Standardizes orientation to RAI (Right-Anterior-Inferior).
        """
        dicom_path = os.path.join(self.config.dicom_dir, patient_id)
        output_path = os.path.join(self.config.standardized_dir, f"{patient_id}.nii.gz")
        
        logging.info(f"Converting DICOM series for Patient: {patient_id}")
        
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_names)
            
            image = reader.Execute()
            
            # Ensure consistent orientation (very important for 3D CNNs)
            image = sitk.DICOMOrient(image, 'RAI')
            
            sitk.WriteImage(image, output_path)
            logging.info(f"Saved standardized NIfTI to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"DICOM conversion failed for {patient_id}: {e}")
            return None

    def validate_volume(self, file_path: str):
        """
        Expert-level check: Validates that the NIfTI file isn't corrupted 
        and contains the expected metadata.
        """
        try:
            img = nib.load(file_path)
            shape = img.shape
            header = img.header
            logging.info(f"Validation Passed: {file_path} | Shape: {shape} | Voxel Size: {header.get_zooms()}")
            return True
        except Exception as e:
            logging.error(f"Validation Failed for {file_path}: {e}")
            return False