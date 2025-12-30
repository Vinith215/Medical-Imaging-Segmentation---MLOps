import torch
from config.model_config import ModelConfig
from config.global_config import GlobalConfig

# Perform MONAI imports lazily inside get_model to avoid import-time errors
# in environments where MONAI is not installed.

class ModelFactory:
    """
    Factory class to generate medical segmentation architectures 
    based on the project configuration.
    """
    
    @staticmethod
    def get_model(model_name: str = None):
        """
        Returns the initialized model specified in the config.
        """
        name = model_name or ModelConfig.MODEL_NAME
        
        if name == "UNet":
            print("🏗️  Building 3D U-Net...")
            try:
                import importlib
                _mod = importlib.import_module("monai.networks.nets")
                UNet = getattr(_mod, "UNet")
            except Exception as e:
                raise ImportError(
                    "MONAI is required to build UNet; install it with 'pip install monai'"
                ) from e

            return UNet(
                spatial_dims=ModelConfig.UNET_SETTINGS["spatial_dims"],
                in_channels=ModelConfig.IN_CHANNELS,
                out_channels=ModelConfig.OUT_CHANNELS,
                channels=ModelConfig.UNET_SETTINGS["channels"],
                strides=ModelConfig.UNET_SETTINGS["strides"],
                num_res_units=ModelConfig.UNET_SETTINGS["num_res_units"],
                norm=ModelConfig.UNET_SETTINGS["norm"],
                dropout=ModelConfig.UNET_SETTINGS["dropout"]
            )
            
        elif name == "SwinUNETR":
            print("🏗️  Building Swin-UNETR (Transformer)...")
            try:
                import importlib
                _mod = importlib.import_module("monai.networks.nets")
                SwinUNETR = getattr(_mod, "SwinUNETR")
            except Exception as e:
                raise ImportError(
                    "MONAI is required to build SwinUNETR; install it with 'pip install monai'"
                ) from e

            return SwinUNETR(
                img_size=ModelConfig.SWIN_UNETR_SETTINGS["img_size"],
                in_channels=ModelConfig.SWIN_UNETR_SETTINGS["in_channels"],
                out_channels=ModelConfig.SWIN_UNETR_SETTINGS["out_channels"],
                feature_size=ModelConfig.SWIN_UNETR_SETTINGS["feature_size"],
                use_checkpoint=ModelConfig.SWIN_UNETR_SETTINGS["use_checkpoint"]
            )
            
        else:
            raise ValueError(f"Model {name} not supported. Choose 'UNet' or 'SwinUNETR'.")

    @staticmethod
    def load_production_model(path: str):
        """
        Loads a model state dict for inference.
        """
        model = ModelFactory.get_model()
        device = torch.device(GlobalConfig.DEVICE)
        
        # Using map_location to ensure it loads on CPU if GPU is unavailable
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

if __name__ == "__main__":
    # Test instantiation
    test_model = ModelFactory.get_model("SwinUNETR")
    print(f"✅ Model generated successfully with {sum(p.numel() for p in test_model.parameters()):,} parameters.")