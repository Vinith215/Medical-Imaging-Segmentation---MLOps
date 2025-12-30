import torch
from monai.networks.nets import SwinUNETR, UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

class SegmentationFactory:
    @staticmethod
    def get_model(config: ModelConfig, device):
        if config.model_type == "SwinUNETR":
            model = SwinUNETR(
                img_size=config.img_size,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                feature_size=48
            ).to(device)
        else:
            model = UNet(
                spatial_dims=3,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
            ).to(device)
        return model

    @staticmethod
    def get_loss_function():
        # DiceLoss is robust to class imbalance (common in medical imaging)
        return DiceLoss(to_onehot_y=True, softmax=True)

    @staticmethod
    def get_evaluation_metric():
        return DiceMetric(include_background=False, reduction="mean")