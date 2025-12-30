import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandRotated,
    RandFlipd,
    ToTensord,
    RandGaussianNoised,
    RandElasticDeformd
)

class MedicalTransformations:
    def __init__(self, config: TransformConfig):
        self.config = config

    def get_train_transforms(self):
        """
        Comprehensive pipeline: Load -> Resample -> Normalize -> Augment -> Patch.
        """
        return Compose([
            # 1. Load and Standardize Metadata
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            
            # 2. Resampling (Physical Standardization)
            # Ensures 1 voxel = 1mm in all directions
            Spacingd(
                keys=["image", "label"], 
                pixdim=self.config.pixdim, 
                mode=("bilinear", "nearest")
            ),
            
            # 3. Intensity Normalization (Windowing)
            # Focuses the model on relevant tissue densities
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=self.config.a_min, a_max=self.config.a_max,
                b_min=self.config.b_min, b_max=self.config.b_max, 
                clip=True
            ),
            
            # 4. Patch Extraction
            # 3D volumes are too big for GPU memory; we take random patches
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.config.roi_size,
                pos=1, neg=1, num_samples=4,
                image_key="image",
                image_threshold=0
            ),
            
            # 5. Data Augmentation (Robustness)
            RandRotated(keys=["image", "label"], range_x=0.3, prob=self.config.prob_rotate),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.config.prob_flip),
            RandGaussianNoised(keys=["image"], prob=0.1),
            
            # 6. Elastic Deformation (Simulates anatomical variation)
            RandElasticDeformd(
                keys=["image", "label"],
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=self.config.prob_elastic,
                spatial_size=self.config.roi_size
            ),
            
            ToTensord(keys=["image", "label"])
        ])

    def get_val_transforms(self):
        """
        Validation transforms do NOT include augmentations.
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=self.config.pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=self.config.a_min, a_max=self.config.a_max, b_min=0, b_max=1, clip=True),
            ToTensord(keys=["image", "label"])
        ])