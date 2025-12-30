import torch
import numpy as np
import importlib
try:
    nib = importlib.import_module("nibabel")
except ImportError as e:
    raise ImportError("nibabel is required for NIfTI IO. Install it with 'pip install nibabel'") from e

try:
    monai_transforms = importlib.import_module("monai.transforms")
    Compose = monai_transforms.Compose
    Activations = monai_transforms.Activations
    AsDiscrete = monai_transforms.AsDiscrete
except ImportError as e:
    raise ImportError("MONAI is required for transforms. Install it with 'pip install monai'") from e

from config.global_config import GlobalConfig
from config.model_config import ModelConfig
from config.transform_config import TransformConfig
from src.model_factory import ModelFactory

class SegmentationInference:
    def __init__(self, model_path: str = None):
        self.device = torch.device(GlobalConfig.DEVICE)
        
        # Load the production model (defaults to production folder if not provided)
        path = model_path or (GlobalConfig.PRODUCTION_MODEL_DIR / "best_model.pth")
        self.model = ModelFactory.load_production_model(path)
        
        # Define inference-specific transforms (No labels expected here)
        self.pre_transforms = TransformConfig.get_inference_transforms()
        
        # Post-processing transforms
        self.post_transforms = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True)
        ])

    def predict(self, image_path: str, output_path: str = None):
        """
        Runs 3D sliding window inference on a NIfTI volume.
        """
        # 1. Prepare data
        data = {"image": image_path}
        processed_data = self.pre_transforms(data)
        # Add batch dimension [1, C, H, W, D]
        input_tensor = processed_data["image"].unsqueeze(0).to(self.device)

        # 2. Execute Sliding Window Inference
        # This prevents OOM (Out of Memory) by processing patches
        print(f"🧠 Running 3D Inference on {image_path}...")
        self.model.eval()
        with torch.no_grad():
            try:
                inferers = importlib.import_module("monai.inferers")
                sliding_window_inference = getattr(inferers, "sliding_window_inference")
            except ImportError as e:
                raise ImportError("MONAI is required for sliding window inference. Install it with 'pip install monai'") from e

            output_logit = sliding_window_inference(
                inputs=input_tensor,
                roi_size=ModelConfig.ROI_SIZE,
                sw_batch_size=ModelConfig.SW_BATCH_SIZE,
                predictor=self.model,
                overlap=ModelConfig.OVERLAP,
                mode="gaussian" # Blends patch edges for smoothness
            )
            
            # 3. Apply Post-processing
            prediction = self.post_transforms(output_logit[0]) # Remove batch dim
            prediction_np = prediction.detach().cpu().numpy()[0] # [H, W, D]

        # 4. Save result to NIfTI
        if output_path:
            # We use the original image header to ensure the mask aligns with the scan
            original_nifti = nib.load(image_path)
            mask_nifti = nib.Nifti1Image(
                prediction_np.astype(np.uint8), 
                original_nifti.affine, 
                original_nifti.header
            )
            nib.save(mask_nifti, output_path)
            print(f"💾 Segmentation saved to: {output_path}")

        return prediction_np

if __name__ == "__main__":
    # Example usage for testing
    test_img = "data/processed/PAT_001_clean.nii.gz"
    test_out = "outputs/PAT_001_mask.nii.gz"
    
    engine = SegmentationInference()
    engine.predict(test_img, test_out)