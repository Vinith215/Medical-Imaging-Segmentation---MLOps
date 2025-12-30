from config.global_config import GlobalConfig

class ModelConfig:
    # --- General Model Settings ---
    MODEL_NAME = "SwinUNETR"  # Options: "UNet", "SwinUNETR"
    IN_CHANNELS = 1           # e.g., 1 for CT/MRI, 3 for RGB
    OUT_CHANNELS = 2          # e.g., 1 background + 1 organ
    
    # --- 3D U-Net Specifics ---
    UNET_SETTINGS = {
        "spatial_dims": 3,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
        "norm": "INSTANCE",   # Medical imaging standard: InstanceNorm
        "dropout": 0.1
    }

    # --- Swin-UNETR Specifics (Transformer-based) ---
    SWIN_UNETR_SETTINGS = {
        "img_size": GlobalConfig.IMAGE_SIZE, # (128, 128, 128)
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "feature_size": 48,  # Size of the first feature map
        "use_checkpoint": True, # For gradient checkpointing (saves VRAM)
    }

    # --- Training Hyperparameters ---
    BATCH_SIZE = 2            # 3D models are heavy on VRAM
    MAX_EPOCHS = 300
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # --- Sliding Window Inference (MONAI) ---
    # This is used in production to process large scans in overlapping chunks
    ROI_SIZE = (128, 128, 128) # Must match GlobalConfig.IMAGE_SIZE
    SW_BATCH_SIZE = 4          # Number of patches processed at once during inference
    OVERLAP = 0.5              # 50% overlap between windows for smooth edges

    # --- Loss & Optimizer Settings ---
    # DiceLoss is the gold standard for segmentation
    # FocalLoss helps if the organ is very small compared to the background
    LOSS_TYPE = "DiceFocalLoss" 
    OPTIMIZER = "AdamW"        # Best for Transformers/Swin-UNETR
    
    # --- Scheduler Settings ---
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = MAX_EPOCHS         # For Cosine Annealing