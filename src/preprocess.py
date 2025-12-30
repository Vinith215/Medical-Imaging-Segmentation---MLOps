import nibabel as nib
import numpy as np
from pathlib import Path
from config.global_config import GlobalConfig
from config.transform_config import TransformConfig

def process_raw_volume(image_path, label_path, output_id):
    """
    Standardizes a raw medical volume for the training pipeline.
    """
    # 1. Define the transformation chain (from our config)
    # We use the 'base' and 'intensity' transforms only
    pre_process = TransformConfig.get_val_transforms()

    # 2. Execute transformation
    data_dict = {"image": str(image_path), "label": str(label_path)}
    processed_data = pre_process(data_dict)

    # 3. Save as NIfTI for visual verification
    out_img_path = GlobalConfig.PROCESSED_DATA_DIR / f"{output_id}_img.nii.gz"
    out_lbl_path = GlobalConfig.PROCESSED_DATA_DIR / f"{output_id}_seg.nii.gz"

    # Convert back from Tensor to NIfTI using MONAI/Nibabel utilities
    # (Simplified for logic flow)
    save_nifti(processed_data["image"], out_img_path)
    save_nifti(processed_data["label"], out_lbl_path)
    
    print(f"✅ Processed and saved: {output_id}")

def save_nifti(tensor, path):
    # Utility to save torch tensor back to NIfTI
    data = tensor.detach().cpu().numpy()[0] # Remove channel dim
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)