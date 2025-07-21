import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Any, Dict, List
import torchvision.transforms as transforms

from src.utils.logging_config import setup_logging
from src.models.base_model import BaseModel, ExplainableBaseModel

setup_logging()
logger = logging.getLogger(__name__)


# --- U-Net Architecture (Simplified) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Simplified U-Net for semantic segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128 + 128, 128)  # Concatenation
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        logger.info("Simplified U-Net architecture initialized.")

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)

        bottleneck = self.bottleneck(pool2)

        up2 = self.upconv2(bottleneck)
        # Center-crop and concatenate if sizes don't match exactly
        diffY2 = enc2.size()[2] - up2.size()[2]
        diffX2 = enc2.size()[3] - up2.size()[3]
        up2 = F.pad(
            up2, [diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2]
        )
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        diffY1 = enc1.size()[2] - up1.size()[2]
        diffX1 = enc1.size()[3] - up1.size()[3]
        up1 = F.pad(
            up1, [diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2]
        )
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        return torch.sigmoid(self.final_conv(dec1))  # Sigmoid for binary segmentation


class SegmentationModel(ExplainableBaseModel):
    """
    Implements an image segmentation model (e.g., for tumor or organ segmentation).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__(model_name="SegmentationModel")
        self._model = UNet(in_channels=in_channels, out_channels=out_channels)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128)),  # Example fixed size for U-Net
                transforms.Normalize(mean=[0.5], std=[0.5]),  # For grayscale input
            ]
        )
        logger.info("SegmentationModel initialized with U-Net.")

    def _load_model_weights(self, model_path: str):
        """Loads PyTorch model state dict."""
        try:
            self._model.load_state_dict(torch.load(model_path))
            self._model.eval()
            logger.info(f"SegmentationModel weights loaded from {model_path}.")
        except Exception as e:
            logger.error(f"Error loading SegmentationModel weights: {e}")
            raise

    def _train_model(self, data: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Trains the segmentation model.
        `data` expects a list of dicts: [{'image': np_array, 'mask': np_array}, ...]
        `config` expects {'epochs': int, 'learning_rate': float}
        """
        self._model.train()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("learning_rate", 0.001)
        )
        criterion = nn.BCELoss()  # Binary Cross-Entropy for binary segmentation

        # Dummy data conversion for training (in real scenario, use DataLoader)
        images_tensor = torch.stack(
            [self.transform(img_data["image"]) for img_data in data]
        )
        masks_tensor = torch.stack(
            [self.transform(img_data["mask"]) for img_data in data]
        )  # Masks should also be transformed

        for epoch in range(config.get("epochs", 10)):
            optimizer.zero_grad()
            outputs = self._model(images_tensor)
            loss = criterion(outputs, masks_tensor)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")

        output_model_path = config.get(
            "output_model_path", "models/segmentation_model.pth"
        )
        torch.save(self._model.state_dict(), output_model_path)
        logger.info(f"SegmentationModel saved to {output_model_path}.")

    def _perform_prediction(self, data: Any) -> Dict[str, Any]:
        """
        Performs segmentation prediction on image data.
        `data` expects a numpy array representing the image.
        Returns a dictionary with 'segmentation_mask' (numpy array, 0-1 range).
        """
        if self._model is None:
            logger.error("Segmentation model is not loaded. Cannot predict.")
            raise RuntimeError("Segmentation model not loaded.")

        self._model.eval()
        with torch.no_grad():
            input_tensor = self.transform(data).unsqueeze(0)  # Add batch dimension
            output_mask = (
                self._model(input_tensor).squeeze(0).cpu().numpy()
            )  # Remove batch dim, to numpy
            logger.debug("SegmentationModel prediction performed.")
        return {
            "segmentation_mask": output_mask.tolist()
        }  # Convert to list for JSON serialization

    def _evaluate_model(self, data: Any) -> Dict[str, Any]:
        """
        Evaluates the model on test data (e.g., using Dice score).
        `data` expects a list of dicts: [{'image': np_array, 'mask': np_array}, ...]
        """
        self._model.eval()
        dice_scores = []
        with torch.no_grad():
            for item in data:
                input_tensor = self.transform(item["image"]).unsqueeze(0)
                true_mask_tensor = self.transform(item["mask"]).unsqueeze(0)
                predicted_mask = self._model(input_tensor)

                # Calculate Dice Score (simplified, assumes binary 0/1 masks)
                intersection = (predicted_mask * true_mask_tensor).sum()
                union = predicted_mask.sum() + true_mask_tensor.sum()
                dice = (2.0 * intersection + 1e-8) / (
                    union + 1e-8
                )  # Add small epsilon for stability
                dice_scores.append(dice.item())

        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        logger.info(f"SegmentationModel evaluation average Dice score: {avg_dice:.4f}")
        return {"average_dice_score": avg_dice}

    def explain(self, data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generates explanation for segmentation, e.g., highlighting contributing regions.
        """
        logger.info(
            "Generating explanation for segmentation (conceptual XAI for U-Net)."
        )
        # For segmentation, explanation could be visualizing the attention maps or
        # intermediate feature maps from the U-Net.
        explanation_map = "Conceptual attention/feature map highlighting relevant image regions for segmentation."
        return {
            "explanation": explanation_map,
            "method": "Conceptual Feature Map Visualization",
        }


if __name__ == "__main__":
    import torch.nn.functional as F  # Need this for F.pad in UNet

    setup_logging()

    # Create dummy image and mask data (128x128 grayscale)
    dummy_image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    # Create a simple dummy mask (e.g., a central square as the target region)
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    dummy_mask[40:80, 40:80] = 1  # A 1-pixel value for the object

    dummy_train_data = [
        {
            "image": np.random.randint(0, 255, size=(128, 128), dtype=np.uint8),
            "mask": (np.random.rand(128, 128) > 0.8).astype(np.uint8),
        }  # Random masks
        for _ in range(5)
    ]
    dummy_eval_data = [
        {
            "image": np.random.randint(0, 255, size=(128, 128), dtype=np.uint8),
            "mask": (np.random.rand(128, 128) > 0.8).astype(np.uint8),
        }  # Random masks
        for _ in range(2)
    ]

    # Initialize and train model (conceptual training)
    segmenter = SegmentationModel(
        in_channels=1, out_channels=1
    )  # Grayscale input, binary mask output
    train_config = {
        "epochs": 2,
        "learning_rate": 0.001,
        "output_model_path": "models/test_segmentation_model.pth",
    }
    segmenter.train(dummy_train_data, train_config)

    # Load the trained model
    segmenter.load("models/test_segmentation_model.pth")

    # Perform prediction
    prediction_result = segmenter.predict(dummy_image)
    print(
        f"Prediction Result (mask shape): {np.array(prediction_result['segmentation_mask']).shape}"
    )
    # print(f"Sample mask pixels: {np.array(prediction_result['segmentation_mask'])[0, 0:5]}")

    # Evaluate model
    eval_metrics = segmenter.evaluate(dummy_eval_data)
    print(f"Evaluation Metrics: {eval_metrics}")

    # Get explanation
    explanation = segmenter.explain(dummy_image, prediction_result)
    print(f"Explanation: {explanation}")
