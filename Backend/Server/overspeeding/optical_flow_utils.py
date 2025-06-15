"""
Optical Flow Utilities for Speed Estimation
Includes preprocessing, visualization, and speed calculation functions
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (220, 66)) -> np.ndarray:
    """
    Preprocess frame for optical flow calculation

    Args:
        frame: Input frame (H, W, 3) or (H, W)
        target_size: Target size (width, height)

    Returns:
        Preprocessed frame
    """
    if frame is None or frame.size == 0:
        # Return a black frame if input is invalid
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    height, width = frame.shape[:2]

    # Ensure we have valid dimensions
    if height <= 0 or width <= 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Simple crop to focus on road area (bottom 2/3 of frame)
    crop_top = height // 3
    crop_bottom = height
    crop_left = 0
    crop_right = width

    # Ensure crop dimensions are valid
    if crop_bottom > crop_top and crop_right > crop_left:
        cropped = frame[crop_top:crop_bottom, crop_left:crop_right]
    else:
        cropped = frame

    # Ensure cropped frame is not empty
    if cropped.size == 0:
        cropped = frame

    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    return resized


def augment_brightness(frame: np.ndarray, brightness_factor: Optional[float] = None) -> np.ndarray:
    """
    Augment frame brightness to handle illumination changes
    
    Args:
        frame: Input frame
        brightness_factor: Brightness factor (if None, random factor is used)
    
    Returns:
        Brightness-augmented frame
    """
    if brightness_factor is None:
        brightness_factor = 0.2 + np.random.uniform(0, 0.8)
    
    # Convert to HSV for brightness adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    
    # Adjust brightness (V channel)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return augmented


def frames_to_tensor(frame1: np.ndarray, frame2: np.ndarray,
                    brightness_factor: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert frames to PyTorch tensors for PWC-Net input

    Args:
        frame1: First frame (H, W, 3) or (H, W)
        frame2: Second frame (H, W, 3) or (H, W)
        brightness_factor: Brightness augmentation factor

    Returns:
        Tuple of tensors (frame1_tensor, frame2_tensor) with shape (3, H, W)
    """
    # Ensure frames are 3-channel
    if len(frame1.shape) == 2:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    if len(frame2.shape) == 2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    # Apply same brightness augmentation to both frames
    if brightness_factor is None:
        brightness_factor = 1.0  # No augmentation by default
    
    frame1_aug = augment_brightness(frame1, brightness_factor)
    frame2_aug = augment_brightness(frame2, brightness_factor)
    
    # Preprocess frames
    frame1_proc = preprocess_frame(frame1_aug)
    frame2_proc = preprocess_frame(frame2_aug)
    
    # Convert to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW format
    ])
    
    tensor1 = transform(frame1_proc)
    tensor2 = transform(frame2_proc)
    
    return tensor1, tensor2


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB visualization
    
    Args:
        flow: Optical flow array (H, W, 2)
    
    Returns:
        RGB visualization of flow
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # Calculate angle and magnitude
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.minimum(v * 4, 255)
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def calculate_speed_from_flow(flow: np.ndarray, fps: float = 30.0, 
                             scale_factor: float = 0.02) -> float:
    """
    Calculate speed from optical flow
    
    Args:
        flow: Optical flow array (H, W, 2)
        fps: Video frame rate
        scale_factor: Calibration factor (pixels per km/h)
    
    Returns:
        Estimated speed in km/h
    """
    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    
    # Use median magnitude to reduce noise
    median_magnitude = np.median(magnitude)
    
    # Convert to speed (this is a simplified approach)
    # In practice, this would need proper calibration
    speed_kmh = median_magnitude / scale_factor * fps / 30.0
    
    return float(speed_kmh)


def visualize_flow_on_frame(frame: np.ndarray, flow: np.ndarray, 
                           step: int = 16) -> np.ndarray:
    """
    Visualize optical flow vectors on frame
    
    Args:
        frame: Original frame
        flow: Optical flow
        step: Step size for flow vector visualization
    
    Returns:
        Frame with flow vectors overlaid
    """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # Create lines for flow vectors
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Draw flow vectors
    vis = frame.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    
    # Draw points
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    
    return vis


def estimate_ego_motion(flow: np.ndarray, method: str = 'median') -> Tuple[float, float]:
    """
    Estimate ego motion (camera/vehicle movement) from optical flow
    
    Args:
        flow: Optical flow array
        method: Method for estimation ('median', 'mean', 'robust')
    
    Returns:
        Tuple of (forward_speed, lateral_speed)
    """
    # Extract flow components
    u = flow[:, :, 0]  # Horizontal flow
    v = flow[:, :, 1]  # Vertical flow
    
    if method == 'median':
        forward_speed = np.median(v)
        lateral_speed = np.median(u)
    elif method == 'mean':
        forward_speed = np.mean(v)
        lateral_speed = np.mean(u)
    elif method == 'robust':
        # Use robust statistics (trimmed mean)
        forward_speed = np.mean(np.sort(v.flatten())[len(v.flatten())//4:-len(v.flatten())//4])
        lateral_speed = np.mean(np.sort(u.flatten())[len(u.flatten())//4:-len(u.flatten())//4])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(forward_speed), float(lateral_speed)


def scale_flow_for_cnn(flow: torch.Tensor, target_size: Tuple[int, int] = (220, 66)) -> torch.Tensor:
    """
    Scale optical flow tensor for CNN input
    
    Args:
        flow: Flow tensor (2, H, W)
        target_size: Target size (width, height)
    
    Returns:
        Scaled flow tensor
    """
    # Resize flow to target size
    flow_resized = torch.nn.functional.interpolate(
        flow.unsqueeze(0), 
        size=(target_size[1], target_size[0]), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # Scale flow values proportionally
    h_scale = target_size[1] / flow.shape[1]
    w_scale = target_size[0] / flow.shape[2]
    
    flow_resized[0] *= w_scale  # Scale horizontal component
    flow_resized[1] *= h_scale  # Scale vertical component
    
    return flow_resized


def create_flow_dataset(frames: list, target_size: Tuple[int, int] = (220, 66)) -> torch.Tensor:
    """
    Create optical flow dataset from list of frames
    
    Args:
        frames: List of frames
        target_size: Target size for flow tensors
    
    Returns:
        Stacked flow tensors
    """
    flow_tensors = []
    
    for i in range(len(frames) - 1):
        frame1, frame2 = frames_to_tensor(frames[i], frames[i + 1])
        
        # Add batch dimension for PWC-Net
        frame1 = frame1.unsqueeze(0)
        frame2 = frame2.unsqueeze(0)
        
        # This would be replaced with actual PWC-Net inference
        # For now, create dummy flow
        dummy_flow = torch.randn(1, 2, target_size[1], target_size[0])
        flow_tensors.append(dummy_flow.squeeze(0))
    
    if flow_tensors:
        return torch.stack(flow_tensors)
    else:
        return torch.empty(0, 2, target_size[1], target_size[0])


def save_flow_visualization(flow: np.ndarray, save_path: str):
    """
    Save optical flow visualization
    
    Args:
        flow: Optical flow array
        save_path: Path to save the visualization
    """
    flow_rgb = flow_to_rgb(flow)
    cv2.imwrite(save_path, cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
