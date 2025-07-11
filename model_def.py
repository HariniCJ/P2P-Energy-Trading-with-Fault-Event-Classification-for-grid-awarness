# model_def.py
# Contains the definition of the PyTorch model architecture.
# This must match the architecture used during training.

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    Defines the 1-D Convolutional Neural Network architecture
    used for classifying power system waveforms.
    """
    def __init__(self, nc):
        """
        Initializes the ConvNet model layers.
        Args:
            nc (int): Number of output classes.
        """
        super().__init__()
        # Feature extractor using Convolutional layers
        self.f = nn.Sequential(
            # Layer 1: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3), # 3 input phases, 32 feature maps
            nn.BatchNorm1d(num_features=32), # Normalize features
            nn.ReLU(), # Activation function
            nn.MaxPool1d(kernel_size=2), # Downsample sequence length by factor of 2

            # Layer 2: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2), # Increase feature maps
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Downsample again

            # Layer 3: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # Increase feature maps further
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Final downsampling
        )

        # Determine the flattened feature dimension dynamically
        # We pass a dummy tensor through the feature extractor to get the output size
        with torch.no_grad(): # No need to track gradients here
            # Create a dummy input tensor with expected dimensions (batch_size=1, channels=3, length=726)
            # Assuming input length is 726 based on training data processing
            dummy_input = torch.zeros(1, 3, 726)
            # Pass the dummy input through the convolutional layers
            feature_output = self.f(dummy_input)
            # Flatten all dimensions except the batch dimension (dim 0)
            # .view(1, -1) reshapes the output into [1, flattened_size]
            feat_dim = feature_output.view(1, -1).shape[1] # Get the size of the flattened dimension

        # Classifier head using Linear layers
        self.cls = nn.Sequential(
            # Fully connected layer 1
            nn.Linear(in_features=feat_dim, out_features=256), # Input features from conv layers, output 256
            nn.ReLU(), # Activation
            nn.Dropout(p=0.4), # Dropout for regularization (prevents overfitting)

            # Output layer
            nn.Linear(in_features=256, out_features=nc) # Final layer maps to number of classes
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 726).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # 1. Pass input through feature extractor (convolutional layers)
        features = self.f(x)
        # 2. Flatten the features before passing to the classifier
        #    Flattens dimensions from 1 onwards (keeps batch dimension)
        flattened_features = features.flatten(start_dim=1)
        # 3. Pass flattened features through the classifier (linear layers)
        output = self.cls(flattened_features)
        return output

