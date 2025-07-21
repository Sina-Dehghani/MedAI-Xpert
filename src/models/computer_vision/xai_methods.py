import torch
import logging
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from typing import Callable, Union, Tuple, Dict, Any

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Dummy class for target_layer in Grad-CAM demonstration
class DummyTargetLayer(torch.nn.Module):
    def __init__(self, target_module: torch.nn.Module):
        super().__init__()
        self.target_module = target_module

    def forward(self, x):
        return self.target_module(x)


class GradCAM:
    """
    Implements a conceptual Grad-CAM (Gradient-weighted Class Activation Mapping) for PyTorch models.
    Note: This is a simplified, conceptual implementation. Real Grad-CAM requires careful
    hooking into specific layers and handling gradients. Libraries like `pytorch-gradcam`
    or `captum` are recommended for robust implementations.
    """

    def __init__(
        self, model: torch.nn.Module, target_layer: Union[str, torch.nn.Module]
    ):
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        # Register hooks to capture gradients and activations
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        if isinstance(target_layer, str):
            # Find the module by name (simplistic, might need recursive search for nested modules)
            modules = dict(self.model.named_modules())
            self.target_module = modules.get(target_layer)
            if not self.target_module:
                logger.error(f"Target layer '{target_layer}' not found in model.")
                raise ValueError(f"Target layer '{target_layer}' not found.")
        else:  # Assume it's a direct torch.nn.Module
            self.target_module = target_layer

        self.target_module.register_forward_hook(save_activation)
        self.target_module.register_backward_hook(save_gradient)

        logger.info(f"Grad-CAM initialized for target layer: {self.target_module}")

    def generate_heatmap(
        self, input_image: torch.Tensor, target_class: int = None
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap.
        `input_image` should be a preprocessed tensor ready for model input.
        `target_class` is the class for which to generate the heatmap. If None, uses predicted class.
        """
        if input_image.dim() == 3:  # Add batch dimension if missing
            input_image = input_image.unsqueeze(0)

        # Ensure input requires gradients
        input_image.requires_grad_(True)

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=-1).item()  # Use predicted class

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        # Create a one-hot vector for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            logger.error(
                "Failed to capture gradients or activations. Hooks might not be firing correctly."
            )
            return np.zeros(input_image.shape[2:])

        # Pool the gradients over all the pixels of the feature map
        # This is the 'weights' for the feature maps
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations and apply ReLU
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize the heatmap to 0-1
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)

        # Resize heatmap to original image size
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = np.maximum(
            heatmap_np, 0
        )  # Apply ReLU again after numpy conversion

        # Simple resizing (real implementation uses more sophisticated interpolation)
        from skimage.transform import resize

        original_size = input_image.shape[2:]  # H, W
        heatmap_resized = resize(heatmap_np, original_size, anti_aliasing=True)

        logger.info(
            f"Grad-CAM heatmap generated for class {target_class}. Shape: {heatmap_resized.shape}"
        )
        return heatmap_resized


if __name__ == "__main__":
    setup_logging()
    from src.models.computer_vision.image_detector import (
        SimpleCNN,
    )  # Import the dummy CNN

    print("\n--- Grad-CAM Example ---")

    # 1. Create a dummy CNN model and load dummy weights
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = SimpleCNN(
                num_classes=10
            ).features  # Use features from SimpleCNN
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Linear(
                32, 10
            )  # 32 is output from SimpleCNN.features' last conv

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    dummy_model = DummyModel()
    # For Grad-CAM, choose a convolutional layer before global pooling
    # In SimpleCNN, 'features.3' (the second Conv2d) or 'features.5' (MaxPool2d after second conv) are good candidates.
    # Let's target the last ConvBlock's output layer for meaningful activations
    # The 'features' module of SimpleCNN has sub-modules. Need to find the correct name/reference.
    # A common target layer name might be 'features.block.1' (the second conv in the last ConvBlock for a deeper model)
    # For SimpleCNN.features, let's target the last conv layer's output (name '1' within block if it was a Sequential(Conv, ReLU))
    # Given SimpleCNN.features:
    # (0): Conv2d
    # (1): ReLU
    # (2): MaxPool2d
    # (3): Conv2d  <-- This is likely the best target for SimpleCNN
    # (4): ReLU
    # (5): MaxPool2d

    target_layer_name = (
        "features.3"  # Name of the last Conv2d layer in SimpleCNN's 'features'
    )

    try:
        # Create a dummy input image (1 channel, 64x64)
        dummy_input_image = torch.randn(1, 1, 64, 64)  # Batch, Channels, H, W

        # Initialize GradCAM
        grad_cam = GradCAM(dummy_model, target_layer=target_layer_name)

        # Generate heatmap for a dummy target class (e.g., class 5)
        heatmap = grad_cam.generate_heatmap(dummy_input_image, target_class=5)

        print(f"Generated heatmap shape: {heatmap.shape}")
        print(f"Heatmap min/max: {heatmap.min():.4f}/{heatmap.max():.4f}")

        # Example: Visualize the heatmap (conceptual, requires matplotlib)
        # import matplotlib.pyplot as plt
        # plt.imshow(dummy_input_image.squeeze().numpy(), cmap='gray')
        # plt.imshow(heatmap, cmap='jet', alpha=0.5) # Overlay heatmap
        # plt.title("Image with Grad-CAM Heatmap")
        # plt.colorbar(label="Activation Strength")
        # plt.show()

    except Exception as e:
        logger.error(f"Error during Grad-CAM example: {e}")
