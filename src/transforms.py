from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import types from MONAI only for static type checking; ignore missing-import diagnostics when monai isn't installed.
    from monai.transforms import (  # type: ignore
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
    # Prefer to import TransformConfig for static checking if available in the package.
    try:
        from .config import TransformConfig  # type: ignore
    except Exception:
        # Provide a minimal stub for static type checkers if the real type isn't available.
        class TransformConfig:  # type: ignore
            pixdim: Any
            a_min: Any
            a_max: Any
            b_min: Any
            b_max: Any
            roi_size: Any
            prob_rotate: Any
            prob_flip: Any
            prob_elastic: Any
else:
    # Runtime fallbacks so the names exist when monai is not installed
    Compose = Any
    LoadImaged = Any
    EnsureChannelFirstd = Any
    Spacingd = Any
    ScaleIntensityRanged = Any
    RandCropByPosNegLabeld = Any
    RandRotated = Any
    RandFlipd = Any
    ToTensord = Any
    RandGaussianNoised = Any
    RandElasticDeformd = Any
    # Ensure TransformConfig name exists at runtime to avoid NameError in annotations;
    # try importing a real implementation, otherwise fall back to Any.
    try:
        from .config import TransformConfig  # type: ignore
    except Exception:
        TransformConfig = Any

def _get_transforms_module():
    try:
        return import_module("monai.transforms")
    except ImportError as e:
        raise ImportError("monai is required for transforms; install with `pip install monai`") from e

class MedicalTransformations:
    def __init__(self, config: TransformConfig):
        self.config = config
        self._mt = _get_transforms_module()

    def get_train_transforms(self):
        """
        Comprehensive pipeline: Load -> Resample -> Normalize -> Augment -> Patch.
        """
        return self._mt.Compose([
            # 1. Load and Standardize Metadata
            self._mt.LoadImaged(keys=["image", "label"]),
            self._mt.EnsureChannelFirstd(keys=["image", "label"]),
            
            # 2. Resampling (Physical Standardization)
            # Ensures 1 voxel = 1mm in all directions
            self._mt.Spacingd(
                keys=["image", "label"], 
                pixdim=self.config.pixdim, 
                mode=("bilinear", "nearest")
            ),
            
            # 3. Intensity Normalization (Windowing)
            # Focuses the model on relevant tissue densities
            self._mt.ScaleIntensityRanged(
                keys=["image"], 
                a_min=self.config.a_min, a_max=self.config.a_max,
                b_min=self.config.b_min, b_max=self.config.b_max, 
                clip=True
            ),
            
            # 4. Patch Extraction
            # 3D volumes are too big for GPU memory; we take random patches
            self._mt.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.config.roi_size,
                pos=1, neg=1, num_samples=4,
                image_key="image",
                image_threshold=0
            ),
            
            # 5. Data Augmentation (Robustness)
            self._mt.RandRotated(keys=["image", "label"], range_x=0.3, prob=self.config.prob_rotate),
            self._mt.RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.config.prob_flip),
            self._mt.RandGaussianNoised(keys=["image"], prob=0.1),
            
            # 6. Elastic Deformation (Simulates anatomical variation)
            self._mt.RandElasticDeformd(
                keys=["image", "label"],
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=self.config.prob_elastic,
                spatial_size=self.config.roi_size
            ),
            
            self._mt.ToTensord(keys=["image", "label"])
        ])

    def get_val_transforms(self):
        """
        Validation transforms do NOT include augmentations.
        """
        return self._mt.Compose([
            self._mt.LoadImaged(keys=["image", "label"]),
            self._mt.EnsureChannelFirstd(keys=["image", "label"]),
            self._mt.Spacingd(keys=["image", "label"], pixdim=self.config.pixdim, mode=("bilinear", "nearest")),
            self._mt.ScaleIntensityRanged(keys=["image"], a_min=self.config.a_min, a_max=self.config.a_max, b_min=0, b_max=1, clip=True),
            self._mt.ToTensord(keys=["image", "label"])
        ])