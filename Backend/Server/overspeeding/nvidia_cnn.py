"""
NVIDIA CNN Architecture for Speed Regression
Based on the End-to-End Deep Learning for Self-Driving Cars approach
Improved with modern techniques and better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NVIDIASpeedNet(nn.Module):
    """
    NVIDIA CNN architecture for speed regression from optical flow
    """
    
    def __init__(self, input_channels=2, dropout_rate=0.5):
        super(NVIDIASpeedNet, self).__init__()
        
        # Normalization layer
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # Convolutional layers (following NVIDIA architecture)
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Calculate the size of flattened features
        # For input size (2, 64, 220) -> after convolutions: (64, 8, 28)
        self.feature_size = 64 * 8 * 28
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        
        # Batch normalization for FC layers
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.bn_fc3 = nn.BatchNorm1d(10)
        
        # Dropout for FC layers
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input optical flow tensor of shape (batch_size, 2, height, width)
        Returns:
            speed: Predicted speed tensor of shape (batch_size,)
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply dropout after conv3
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with batch normalization and dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc_dropout(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc_dropout(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc_dropout(x)
        
        # Final output (no activation for regression)
        speed = self.fc4(x)
        
        return speed.squeeze(1)  # Remove the last dimension


class ImprovedNVIDIASpeedNet(nn.Module):
    """
    Improved version of NVIDIA CNN with residual connections and attention
    """
    
    def __init__(self, input_channels=2, dropout_rate=0.3):
        super(ImprovedNVIDIASpeedNet, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # Convolutional blocks with residual connections
        self.conv_block1 = self._make_conv_block(input_channels, 24, stride=2)
        self.conv_block2 = self._make_conv_block(24, 36, stride=2)
        self.conv_block3 = self._make_conv_block(36, 48, stride=2)
        self.conv_block4 = self._make_conv_block(48, 64, stride=1)
        self.conv_block5 = self._make_conv_block(64, 64, stride=1)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(10, 1)
        )
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        """Create a convolutional block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with attention mechanism"""
        x = self.input_norm(x)
        
        # Convolutional feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Fully connected layers
        speed = self.fc_layers(x)
        
        return speed.squeeze(1)


def create_speed_model(model_type='nvidia', input_channels=2, dropout_rate=0.5):
    """
    Factory function to create speed estimation models
    
    Args:
        model_type: 'nvidia' or 'improved'
        input_channels: Number of input channels (2 for optical flow)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        PyTorch model
    """
    if model_type == 'nvidia':
        return NVIDIASpeedNet(input_channels, dropout_rate)
    elif model_type == 'improved':
        return ImprovedNVIDIASpeedNet(input_channels, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
