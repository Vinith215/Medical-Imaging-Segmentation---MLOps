from monai import transforms
from config.global_config import GlobalConfig

class TransformConfig:
    """
    Defines the MONAI transformation pipelines for training, 
    validation, and production inference.
    """

    # --- Basic Preprocessing (Shared across Train/Val/Inference) ---
    # 1. Load the NIfTI data
    # 2. Add a channel dimension (C, H, W, D)
    # 3. Orient to Ras (Right, Anterior, Superior)
    # 4. Resample to 1.0mm isotropic spacing
    BASE_TRANSFORMS = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"], 
            pixdim=GlobalConfig.TARGET_SPACING, 
            mode=("bilinear", "nearest")
        ),
    ]

    # --- Intensity Normalization ---
    # For CT: Scales based on Hounsfield Units (HU)
    # For MRI: Scales based on mean/std or percentiles
    INTENSITY_TRANSFORMS = [
        transforms.ScaleIntensityRanged(
            keys=["image"], 
            a_min=GlobalConfig.HU_MIN, 
            a_max=GlobalConfig.HU_MAX,
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    # --- Training Augmentation (Help model generalize) ---
    TRAIN_AUGMENTATION = [
        # Randomly crop a patch of the fixed size defined in global_config
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=GlobalConfig.IMAGE_SIZE,
            random_center=True,
            random_size=False
        ),
        # Randomly flip across all axes
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        # Random rotation and scaling
        transforms.RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.2),
        # Intensity noise
        transforms.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
    ]

    # --- Final Step: Convert to PyTorch Tensors ---
    ENSURE_TENSORS = [transforms.EnsureTyped(keys=["image", "label"])]

    @classmethod
    def get_train_transforms(cls):
        return transforms.Compose(
            cls.BASE_TRANSFORMS + 
            cls.INTENSITY_TRANSFORMS + 
            cls.TRAIN_AUGMENTATION + 
            cls.ENSURE_TENSORS
        )

    @classmethod
    def get_val_transforms(cls):
        # Validation doesn't need random augmentations
        return transforms.Compose(
            cls.BASE_TRANSFORMS + 
            cls.INTENSITY_TRANSFORMS + 
            cls.ENSURE_TENSORS
        )

    @classmethod
    def get_inference_transforms(cls):
        # Inference usually doesn't have a label key in the input dictionary
        inference_base = [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image"], 
                pixdim=GlobalConfig.TARGET_SPACING, 
                mode="bilinear"
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], 
                a_min=GlobalConfig.HU_MIN, 
                a_max=GlobalConfig.HU_MAX,
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            ),
            transforms.EnsureTyped(keys=["image"])
        ]
        return transforms.Compose(inference_base)