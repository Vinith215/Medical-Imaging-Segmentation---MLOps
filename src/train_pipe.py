import os
import json
import torch
import mlflow
from tqdm import tqdm
from monai.data import DataLoader, Dataset  # type: ignore
try:
    from monai.losses import DiceFocalLoss  # type: ignore
except Exception:
    # Fallback: use DiceLoss if DiceFocalLoss is unavailable in the installed MONAI
    from monai.losses import DiceLoss as DiceFocalLoss  # type: ignore

# Try importing MONAI's DiceMetric, otherwise provide a minimal fallback implementation
try:
    from monai.metrics import DiceMetric  # type: ignore
except Exception:
    # Minimal fallback DiceMetric to provide required interface (callable, aggregate, reset).
    class DiceMetric:
        def __init__(self, include_background=False, reduction="mean"):
            self.include_background = include_background
            self.reduction = reduction
            self._scores = []

        def __call__(self, y_pred=None, y=None, *args, **kwargs):
            # Expect y_pred shape (B,1,...) from argmax and y shape (B,1,...)
            pred = y_pred.squeeze(1).float()
            truth = y.squeeze(1).float()
            batch_scores = []
            for p, t in zip(pred, truth):
                p_flat = (p > 0).float().view(-1)
                t_flat = (t > 0).float().view(-1)
                inter = (p_flat * t_flat).sum()
                denom = p_flat.sum() + t_flat.sum()
                score = (2.0 * inter) / (denom + 1e-8)
                batch_scores.append(score.item())
            self._scores.extend(batch_scores)

        def aggregate(self):
            if len(self._scores) == 0:
                return torch.tensor(0.0)
            return torch.tensor(sum(self._scores) / len(self._scores))

        def reset(self):
            self._scores = []

try:
    from monai.handlers.utils import from_engine  # type: ignore
except Exception:
    # Fallback no-op: some MONAI versions may not expose from_engine; provide a simple passthrough.
    def from_engine(x):
        return x

from config.global_config import GlobalConfig
from config.model_config import ModelConfig
from config.transform_config import TransformConfig
from src.model_factory import ModelFactory

class TrainingPipeline:
    def __init__(self):
        self.device = torch.device(GlobalConfig.DEVICE)
        self.model = ModelFactory.get_model().to(self.device)
        
        # Loss: Dice + Focal is robust against class imbalance in medical scans
        self.loss_function = DiceFocalLoss(to_onehot_y=ModelConfig.OUT_CHANNELS, softmax=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=ModelConfig.LEARNING_RATE)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Load splits from Phase 2
        with open(GlobalConfig.METADATA_DIR / "train_val_test_split.json", "r") as f:
            self.splits = json.load(f)

    def prepare_data(self):
        """Prepares MONAI Datasets and DataLoaders."""
        # Helper to get file paths from IDs
        def get_files(id_list):
            return [
                {
                    "image": str(GlobalConfig.PROCESSED_DATA_DIR / f"{pid}_clean.nii.gz"),
                    "label": str(GlobalConfig.PROCESSED_DATA_DIR / f"{pid}_seg.nii.gz") # Assumes masks exist
                } for pid in id_list
            ]

        train_files = get_files(self.splits["train"])
        val_files = get_files(self.splits["val"])

        train_ds = Dataset(data=train_files, transform=TransformConfig.get_train_transforms())
        val_ds = Dataset(data=val_files, transform=TransformConfig.get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=ModelConfig.BATCH_SIZE, shuffle=True, num_workers=GlobalConfig.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=GlobalConfig.NUM_WORKERS)

        return train_loader, val_loader

    def run_training(self):
        train_loader, val_loader = self.prepare_data()
        
        mlflow.set_tracking_uri(GlobalConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(GlobalConfig.EXPERIMENT_NAME)

        with mlflow.start_run():
            # Log Hyperparameters
            mlflow.log_params(ModelConfig.SWIN_UNETR_SETTINGS if ModelConfig.MODEL_NAME == "SwinUNETR" else ModelConfig.UNET_SETTINGS)
            mlflow.log_param("lr", ModelConfig.LEARNING_RATE)

            best_metric = -1
            for epoch in range(ModelConfig.MAX_EPOCHS):
                self.model.train()
                epoch_loss = 0
                step = 0
                
                for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    step += 1
                    inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / step
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                # Validation Phase
                self.model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)
                        # Use sliding window inference for full volumes
                        val_outputs = torch.argmax(self.model(val_inputs), dim=1, keepdim=True)
                        self.dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = self.dice_metric.aggregate().item()
                    self.dice_metric.reset()
                    mlflow.log_metric("val_dice", metric, step=epoch)

                    # Save Checkpoint
                    if metric > best_metric:
                        best_metric = metric
                        checkpoint_path = GlobalConfig.CHECKPOINT_DIR / "best_metric_model.pth"
                        torch.save(self.model.state_dict(), checkpoint_path)
                        mlflow.log_artifact(str(checkpoint_path))
                        print(f"🌟 New Best Metric: {best_metric:.4f} at epoch {epoch}")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_training()