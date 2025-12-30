import pandas as pd
import numpy as np
try:
    import SimpleITK as sitk  # type: ignore[import]
except Exception:
    sitk = None
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

from config.global_config import GlobalConfig
from config.transform_config import TransformConfig

class MedicalPreprocessor:
    def __init__(self):
        if sitk is None:
            raise ImportError("SimpleITK is required. Install it with: pip install SimpleITK")
        self.manifest_path = GlobalConfig.METADATA_DIR / "dataset_manifest.csv"
        self.output_dir = GlobalConfig.PROCESSED_DATA_DIR
        
    def resample_volume(self, itk_image, is_label=False):
        """Resamples an image to the target isotropic spacing defined in GlobalConfig."""
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        
        target_spacing = GlobalConfig.TARGET_SPACING
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        
        # Use Nearest Neighbor for masks to preserve labels, Linear for images
        resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        
        return resample.Execute(itk_image)

    def run_pipeline(self):
        """Processes all volumes listed in the manifest."""
        if not self.manifest_path.exists():
            print("❌ Manifest not found. Run ingestion first.")
            return

        df = pd.read_csv(self.manifest_path)
        processed_records = []

        print("🔄 Starting Preprocessing: Resampling & Normalization...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # 1. Load Image and Label (if exists)
                img_path = row['processed_path']
                itk_img = sitk.ReadImage(img_path)
                
                # 2. Resample to Isotropic Spacing (e.g., 1.0mm)
                resampled_img = self.resample_volume(itk_img, is_label=False)
                
                # 3. Save processed version
                patient_id = row['patient_id']
                out_img_name = f"{patient_id}_clean.nii.gz"
                out_img_path = self.output_dir / out_img_name
                sitk.WriteImage(resampled_img, str(out_img_path))
                
                processed_records.append({
                    "patient_id": patient_id,
                    "processed_img": str(out_img_path),
                    "spacing": GlobalConfig.TARGET_SPACING
                })
                
            except Exception as e:
                print(f"⚠️ Error processing {row['patient_id']}: {e}")

        self.create_splits(processed_records)

    def create_splits(self, records):
        """Creates a reproducible train/val/test split and saves to metadata."""
        df = pd.DataFrame(records)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=GlobalConfig.SEED)
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=GlobalConfig.SEED)
        
        splits = {
            "train": train_df['patient_id'].tolist(),
            "val": val_df['patient_id'].tolist(),
            "test": test_df['patient_id'].tolist()
        }
        
        split_path = GlobalConfig.METADATA_DIR / "train_val_test_split.json"
        with open(split_path, 'w') as f:
            json.dump(splits, f, indent=4)
        
        print(f"✅ Preprocessing complete. Splits saved to {split_path}")

if __name__ == "__main__":
    preprocessor = MedicalPreprocessor()
    preprocessor.run_pipeline()