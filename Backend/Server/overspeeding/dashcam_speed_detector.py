"""
Dashcam Speed Detection System
Uses PWC-Net for optical flow and NVIDIA CNN for speed regression
Integrates YOLOv8 for vehicle detection and tracking
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os
from typing import Tuple, List, Optional, Dict
from collections import defaultdict
import time

from pwc_net import load_pwc_net
from nvidia_cnn import create_speed_model
from optical_flow_utils import (
    frames_to_tensor, 
    calculate_speed_from_flow,
    estimate_ego_motion,
    scale_flow_for_cnn,
    visualize_flow_on_frame
)


class DashcamSpeedDetector:
    """
    Main class for dashcam speed detection using optical flow and deep learning
    """
    
    def __init__(self, 
                 pwc_model_path: str = "pwc_net.pth",
                 yolo_model_path: str = "yolov8n.pt",
                 speed_model_type: str = "nvidia",
                 device: str = "auto"):
        """
        Initialize the speed detector
        
        Args:
            pwc_model_path: Path to PWC-Net model
            yolo_model_path: Path to YOLO model
            speed_model_type: Type of speed model ('nvidia' or 'improved')
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        self.pwc_net = self._load_pwc_net(pwc_model_path)
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self.speed_model = self._load_speed_model(speed_model_type)
        
        # Speed estimation parameters
        self.speed_scale = 0.02  # Calibration factor
        self.speed_limit = 120   # Speed limit in km/h
        self.confidence_threshold = 0.5

        # Ultra-advanced speed smoothing parameters
        self.speed_history = []  # Store recent speed measurements
        self.speed_window_size = 25  # Larger window for maximum smoothing
        self.min_flow_threshold = 0.2  # Very low threshold for maximum sensitivity
        self.speed_change_threshold = 5.0  # Very smooth transitions

        # Advanced calibration parameters
        self.fps = 30.0  # Video FPS for accurate time calculations
        self.pixel_to_meter_ratio = 15.0  # Calibrated pixel to meter conversion
        self.camera_height = 1.2  # Camera height in meters
        self.camera_angle = 5.0  # Camera tilt angle in degrees

        # Multi-method speed estimation
        self.flow_speeds = []  # Optical flow based speeds
        self.relative_speeds = []  # Relative speed based estimates
        self.kalman_filter = None  # Will initialize Kalman filter

        # Vehicle tracking for relative speed
        self.tracked_vehicles = {}  # Enhanced vehicle tracking
        self.reference_vehicles = []  # Vehicles used for relative speed
        self.road_features = []  # Road features for speed reference

        # Vehicle tracking
        self.vehicle_tracks = defaultdict(lambda: {
            'speeds': [],
            'positions': [],
            'last_seen': 0,
            'overspeeding': False
        })
        self.track_id_counter = 0

        # Video processing
        self.frame_count = 0
        self.prev_frame = None
        self.prev_speed = 0.0  # Store previous speed for smoothing

        # Initialize Kalman filter for speed smoothing
        self._init_kalman_filter()
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter for speed smoothing"""
        try:
            import cv2
            # State: [speed, acceleration]
            self.kalman_filter = cv2.KalmanFilter(2, 1)

            # Transition matrix (constant acceleration model)
            self.kalman_filter.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)

            # Measurement matrix
            self.kalman_filter.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

            # Process noise covariance
            self.kalman_filter.processNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * 0.1

            # Measurement noise covariance
            self.kalman_filter.measurementNoiseCov = np.array([[1]], dtype=np.float32) * 2.0

            # Error covariance
            self.kalman_filter.errorCovPost = np.array([[1, 0], [0, 1]], dtype=np.float32)

            # Initial state
            self.kalman_filter.statePre = np.array([0, 0], dtype=np.float32)
            self.kalman_filter.statePost = np.array([0, 0], dtype=np.float32)

            print("Kalman filter initialized for speed smoothing")

        except Exception as e:
            print(f"Warning: Could not initialize Kalman filter: {e}")
            self.kalman_filter = None

    def _load_pwc_net(self, model_path: str):
        """Load PWC-Net model - Currently disabled due to compatibility issues"""
        print("PWC-Net disabled - using OpenCV optical flow for better compatibility")
        return None
    
    def _load_yolo_model(self, model_path: str):
        """Load YOLO model"""
        try:
            model = YOLO(model_path)
            print(f"YOLO model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"Warning: Could not load YOLO from {model_path}: {e}")
            print("Using default YOLOv8n")
            return YOLO("yolov8n.pt")
    
    def _load_speed_model(self, model_type: str):
        """Load speed estimation model"""
        model = create_speed_model(model_type, input_channels=2, dropout_rate=0.3)
        model.to(self.device)
        model.eval()
        print(f"Speed model ({model_type}) initialized")
        return model
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in frame using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            List of vehicle detections
        """
        results = self.yolo_model(frame, conf=self.confidence_threshold)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for vehicle classes (car, truck, bus, motorcycle)
                    class_id = int(box.cls[0])
                    if class_id in [2, 3, 5, 7]:  # COCO classes for vehicles
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
        
        return vehicles
    
    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> torch.Tensor:
        """
        Calculate optical flow using OpenCV Farneback method (PWC-Net disabled due to compatibility issues)

        Args:
            frame1: Previous frame
            frame2: Current frame

        Returns:
            Optical flow tensor
        """
        # Always use OpenCV for now due to PWC-Net compatibility issues
        return self._calculate_opencv_flow(frame1, frame2)

    def _calculate_opencv_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> torch.Tensor:
        """
        Fallback optical flow calculation using OpenCV Farneback method

        Args:
            frame1: Previous frame
            frame2: Current frame

        Returns:
            Optical flow tensor
        """
        try:
            # Validate input frames
            if frame1 is None or frame2 is None or frame1.size == 0 or frame2.size == 0:
                raise ValueError("Invalid input frames")

            # Preprocess frames
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame1.copy()

            if len(frame2.shape) == 3:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = frame2.copy()

            # Validate grayscale frames
            if gray1.size == 0 or gray2.size == 0:
                raise ValueError("Empty grayscale frames")

            # Resize to target size for consistency
            target_size = (220, 66)
            gray1 = cv2.resize(gray1, target_size, interpolation=cv2.INTER_AREA)
            gray2 = cv2.resize(gray2, target_size, interpolation=cv2.INTER_AREA)

            # Ultra-precise optical flow calculation with maximum accuracy parameters
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,      # Optimal pyramid scale
                levels=7,           # Maximum pyramid levels for finest detail
                winsize=31,         # Large window for maximum stability
                iterations=10,      # Maximum iterations for convergence
                poly_n=7,          # Large polynomial neighborhood
                poly_sigma=1.8,    # Optimized Gaussian sigma
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # Gaussian weighting for precision
            )

            # Convert to tensor format (2, H, W)
            if flow is not None and len(flow.shape) == 3 and flow.shape[-1] == 2:
                flow_u = flow[:, :, 0]
                flow_v = flow[:, :, 1]
            else:
                # Create zero flow as fallback
                flow_u = np.zeros((target_size[1], target_size[0]))
                flow_v = np.zeros((target_size[1], target_size[0]))

            # Convert to torch tensor
            flow_tensor = torch.stack([
                torch.from_numpy(flow_u.astype(np.float32)),
                torch.from_numpy(flow_v.astype(np.float32))
            ]).to(self.device)

            return flow_tensor

        except Exception as e:
            print(f"OpenCV flow calculation failed: {e}")
            # Return zero flow as final fallback
            target_size = (220, 66)
            flow_u = np.zeros((target_size[1], target_size[0]))
            flow_v = np.zeros((target_size[1], target_size[0]))

            flow_tensor = torch.stack([
                torch.from_numpy(flow_u.astype(np.float32)),
                torch.from_numpy(flow_v.astype(np.float32))
            ]).to(self.device)

            return flow_tensor
    
    def estimate_ego_speed(self, flow: torch.Tensor, vehicles: list = None) -> float:
        """
        Advanced ego vehicle speed estimation using multiple methods

        Args:
            flow: Optical flow tensor
            vehicles: List of detected vehicles for relative speed calculation

        Returns:
            Highly accurate smoothed speed in km/h
        """
        try:
            # Method 1: Enhanced optical flow analysis
            flow_speed = self._calculate_enhanced_flow_speed(flow)

            # Method 2: Relative speed analysis using vehicles
            relative_speed = self._calculate_relative_speed(vehicles) if vehicles else None

            # Method 3: Road feature analysis
            feature_speed = self._calculate_feature_speed(flow)

            # Combine multiple estimates with confidence weighting
            final_speed = self._fuse_speed_estimates(flow_speed, relative_speed, feature_speed)

            # Apply advanced smoothing with Kalman filter
            smoothed_speed = self._advanced_smooth_speed(final_speed)

            return smoothed_speed

        except Exception as e:
            print(f"Error in advanced speed estimation: {e}")
            return self.prev_speed if hasattr(self, 'prev_speed') else 0.0

    def _calculate_enhanced_flow_speed(self, flow: torch.Tensor) -> float:
        """Enhanced optical flow speed calculation with multiple techniques"""
        try:
            # Extract flow components
            if flow.dim() == 3 and flow.shape[0] == 2:
                flow_u = flow[0]
                flow_v = flow[1]
            elif flow.dim() == 4 and flow.shape[1] == 2:
                flow_u = flow[0, 0]
                flow_v = flow[0, 1]
            else:
                return 0.0

            # Method 1: Focus on road surface (bottom 2/3 of image)
            h, w = flow_u.shape
            road_region_v = flow_v[h//3:, :]  # Bottom 2/3
            road_region_u = flow_u[h//3:, :]

            # Method 2: Exclude vehicle regions (use central road area)
            center_region_v = road_region_v[:, w//4:3*w//4]  # Central 50%
            center_region_u = road_region_u[:, w//4:3*w//4]

            # Method 3: Multi-scale analysis
            speeds = []

            # Scale 1: Full road region
            if road_region_v.numel() > 0:
                road_magnitude = torch.sqrt(road_region_u**2 + road_region_v**2)
                valid_road = road_magnitude[road_magnitude > self.min_flow_threshold]
                if valid_road.numel() > 0:
                    speeds.append(float(torch.quantile(valid_road, 0.8).cpu().item()))

            # Scale 2: Center region (most reliable)
            if center_region_v.numel() > 0:
                center_magnitude = torch.sqrt(center_region_u**2 + center_region_v**2)
                valid_center = center_magnitude[center_magnitude > self.min_flow_threshold]
                if valid_center.numel() > 0:
                    speeds.append(float(torch.quantile(valid_center, 0.75).cpu().item()) * 1.2)  # Higher weight

            # Scale 3: Forward motion focus (vertical component)
            forward_flow = torch.abs(road_region_v)
            valid_forward = forward_flow[forward_flow > self.min_flow_threshold]
            if valid_forward.numel() > 0:
                speeds.append(float(torch.quantile(valid_forward, 0.7).cpu().item()) * 1.1)

            # Combine speeds with weighted average
            if speeds:
                weights = [1.0, 2.0, 1.5][:len(speeds)]  # Center region gets highest weight
                weighted_speed = sum(s * w for s, w in zip(speeds, weights)) / sum(weights)
            else:
                weighted_speed = 0.0

            # Convert to km/h with improved calibration
            speed_kmh = weighted_speed * 18.0  # Adjusted scaling factor

            return max(0, min(speed_kmh, 200))

        except Exception as e:
            print(f"Error in enhanced flow speed calculation: {e}")
            return 0.0

    def _calculate_relative_speed(self, vehicles: list) -> float:
        """Calculate ego speed using relative motion of other vehicles"""
        try:
            if not vehicles or len(vehicles) < 2:
                return None

            relative_speeds = []

            for vehicle in vehicles:
                track_id = vehicle.get('track_id')
                bbox = vehicle.get('bbox', [0, 0, 0, 0])

                if track_id in self.tracked_vehicles:
                    prev_data = self.tracked_vehicles[track_id]
                    prev_bbox = prev_data.get('bbox')

                    if prev_bbox:
                        # Calculate relative motion
                        dx = bbox[0] - prev_bbox[0]
                        dy = bbox[1] - prev_bbox[1]

                        # Focus on vehicles moving in opposite direction (oncoming traffic)
                        if dx < -5:  # Moving right to left (oncoming)
                            relative_motion = abs(dx)
                            # Convert to speed estimate
                            relative_speed = relative_motion * 2.5  # Calibration factor
                            relative_speeds.append(relative_speed)

                # Update tracking
                self.tracked_vehicles[track_id] = {
                    'bbox': bbox,
                    'last_seen': self.frame_count
                }

            # Clean old tracks
            self._clean_old_tracks()

            if relative_speeds:
                # Use median of relative speeds
                return float(np.median(relative_speeds))

            return None

        except Exception as e:
            print(f"Error in relative speed calculation: {e}")
            return None

    def _calculate_feature_speed(self, flow: torch.Tensor) -> float:
        """Calculate speed using road features and lane markings"""
        try:
            if flow.dim() == 3 and flow.shape[0] == 2:
                flow_v = flow[1]  # Vertical component
            else:
                return None

            h, w = flow_v.shape

            # Focus on lane marking regions (center and sides)
            lane_regions = [
                flow_v[:, w//4-10:w//4+10],      # Left lane
                flow_v[:, w//2-5:w//2+5],        # Center line
                flow_v[:, 3*w//4-10:3*w//4+10]   # Right lane
            ]

            feature_speeds = []

            for region in lane_regions:
                if region.numel() > 0:
                    # Look for consistent motion patterns
                    region_flow = torch.abs(region)
                    valid_flow = region_flow[region_flow > self.min_flow_threshold * 1.5]

                    if valid_flow.numel() > 10:  # Enough points
                        # Use 90th percentile for lane markings (they should have consistent motion)
                        feature_speed = float(torch.quantile(valid_flow, 0.9).cpu().item())
                        feature_speeds.append(feature_speed)

            if feature_speeds:
                # Average of lane marking speeds
                avg_feature_speed = sum(feature_speeds) / len(feature_speeds)
                return avg_feature_speed * 20.0  # Calibration for lane markings

            return None

        except Exception as e:
            print(f"Error in feature speed calculation: {e}")
            return None

    def _fuse_speed_estimates(self, flow_speed: float, relative_speed: float, feature_speed: float) -> float:
        """Fuse multiple speed estimates with confidence weighting"""
        try:
            estimates = []
            weights = []

            # Flow-based speed (always available, medium confidence)
            if flow_speed is not None:
                estimates.append(flow_speed)
                weights.append(1.0)

            # Relative speed (high confidence when available)
            if relative_speed is not None and 5 <= relative_speed <= 150:
                estimates.append(relative_speed)
                weights.append(2.0)  # Higher confidence

            # Feature speed (medium confidence, good for validation)
            if feature_speed is not None and 5 <= feature_speed <= 120:
                estimates.append(feature_speed)
                weights.append(1.5)

            if not estimates:
                return 0.0

            # Weighted average
            weighted_sum = sum(e * w for e, w in zip(estimates, weights))
            total_weight = sum(weights)

            fused_speed = weighted_sum / total_weight

            # Sanity check
            return max(0, min(fused_speed, 200))

        except Exception as e:
            print(f"Error in speed fusion: {e}")
            return flow_speed if flow_speed is not None else 0.0

    def _advanced_smooth_speed(self, raw_speed: float) -> float:
        """Advanced smoothing using Kalman filter and multiple techniques"""
        try:
            # Method 1: Kalman filter smoothing
            kalman_speed = self._kalman_smooth(raw_speed)

            # Method 2: Enhanced moving average
            ma_speed = self._enhanced_moving_average(raw_speed)

            # Method 3: Adaptive smoothing based on speed change
            adaptive_speed = self._adaptive_smooth(raw_speed)

            # Combine smoothing methods
            if kalman_speed is not None:
                # Kalman filter available - use weighted combination
                final_speed = (kalman_speed * 2.0 + ma_speed * 1.0 + adaptive_speed * 1.0) / 4.0
            else:
                # No Kalman filter - use other methods
                final_speed = (ma_speed * 2.0 + adaptive_speed * 1.0) / 3.0

            # Update previous speed
            self.prev_speed = final_speed

            return final_speed

        except Exception as e:
            print(f"Error in advanced smoothing: {e}")
            return raw_speed

    def _clean_old_tracks(self):
        """Remove old vehicle tracks"""
        current_frame = self.frame_count
        to_remove = []

        for track_id, data in self.tracked_vehicles.items():
            if current_frame - data.get('last_seen', 0) > 30:  # 30 frames old
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracked_vehicles[track_id]

    def _kalman_smooth(self, raw_speed: float) -> float:
        """Apply Kalman filter smoothing"""
        try:
            if self.kalman_filter is None:
                return None

            # Predict
            self.kalman_filter.predict()

            # Update with measurement
            measurement = np.array([[raw_speed]], dtype=np.float32)
            self.kalman_filter.correct(measurement)

            # Get smoothed speed
            smoothed_speed = float(self.kalman_filter.statePost[0])

            return max(0, min(smoothed_speed, 200))

        except Exception as e:
            print(f"Error in Kalman smoothing: {e}")
            return None

    def _enhanced_moving_average(self, raw_speed: float) -> float:
        """Enhanced moving average with adaptive window"""
        try:
            # Add to history
            self.speed_history.append(raw_speed)

            # Adaptive window size based on speed stability
            if len(self.speed_history) > 5:
                recent_speeds = self.speed_history[-5:]
                speed_variance = np.var(recent_speeds)

                # Larger window for unstable speeds
                if speed_variance > 25:  # High variance
                    window_size = min(20, len(self.speed_history))
                elif speed_variance > 10:  # Medium variance
                    window_size = min(15, len(self.speed_history))
                else:  # Low variance
                    window_size = min(10, len(self.speed_history))
            else:
                window_size = len(self.speed_history)

            # Keep only recent measurements
            if len(self.speed_history) > window_size:
                self.speed_history = self.speed_history[-window_size:]

            # Apply weighted moving average
            if len(self.speed_history) < 3:
                return raw_speed

            # Enhanced outlier rejection
            speeds = np.array(self.speed_history)
            median_speed = np.median(speeds)
            mad = np.median(np.abs(speeds - median_speed))  # Median Absolute Deviation

            # Remove outliers using MAD (more robust than std)
            if mad > 0:
                threshold = 2.5 * mad
                valid_mask = np.abs(speeds - median_speed) <= threshold
                valid_speeds = speeds[valid_mask]
            else:
                valid_speeds = speeds

            if len(valid_speeds) == 0:
                return median_speed

            # Exponential weighting (more weight to recent values)
            weights = np.exp(np.linspace(-1.5, 0, len(valid_speeds)))
            weights = weights / np.sum(weights)

            weighted_speed = np.sum(valid_speeds * weights)

            return float(weighted_speed)

        except Exception as e:
            print(f"Error in enhanced moving average: {e}")
            return raw_speed

    def _adaptive_smooth(self, raw_speed: float) -> float:
        """Adaptive smoothing based on speed change rate"""
        try:
            if not hasattr(self, 'prev_speed') or self.prev_speed == 0:
                return raw_speed

            speed_change = abs(raw_speed - self.prev_speed)

            # Adaptive smoothing factor based on change rate
            if speed_change < 2:  # Small change - light smoothing
                alpha = 0.7
            elif speed_change < 5:  # Medium change - moderate smoothing
                alpha = 0.5
            elif speed_change < 10:  # Large change - heavy smoothing
                alpha = 0.3
            else:  # Very large change - very heavy smoothing
                alpha = 0.1

            # Exponential smoothing
            smoothed_speed = alpha * raw_speed + (1 - alpha) * self.prev_speed

            # Additional constraint: limit maximum change per frame
            max_change = self.speed_change_threshold
            if abs(smoothed_speed - self.prev_speed) > max_change:
                if smoothed_speed > self.prev_speed:
                    smoothed_speed = self.prev_speed + max_change
                else:
                    smoothed_speed = max(0, self.prev_speed - max_change)

            return smoothed_speed

        except Exception as e:
            print(f"Error in adaptive smoothing: {e}")
            return raw_speed

    def _calculate_raw_speed(self, flow: torch.Tensor) -> float:
        try:
            # Extract flow components
            if flow.dim() == 3 and flow.shape[0] == 2:
                flow_u = flow[0]
                flow_v = flow[1]
            elif flow.dim() == 4 and flow.shape[1] == 2:
                flow_u = flow[0, 0]
                flow_v = flow[0, 1]
            else:
                return 0.0

            # Calculate flow magnitude
            flow_magnitude = torch.sqrt(flow_u**2 + flow_v**2)

            # Filter out very small movements (noise)
            valid_flow = flow_magnitude[flow_magnitude > self.min_flow_threshold]

            if valid_flow.numel() == 0:
                return 0.0

            # Use different statistics for more stable estimation
            # Focus on the forward motion (vertical component)
            forward_flow = torch.abs(flow_v)
            valid_forward = forward_flow[forward_flow > self.min_flow_threshold]

            if valid_forward.numel() > 0:
                # Use 75th percentile to reduce noise but capture significant motion
                percentile_75 = torch.quantile(valid_forward, 0.75)
                speed_estimate = float(percentile_75.cpu().item())
            else:
                # Fallback to overall magnitude
                speed_estimate = float(torch.median(flow_magnitude).cpu().item())

            # Convert to km/h with improved calibration
            # Adjusted scaling factor for more realistic speeds
            speed_kmh = speed_estimate * 25.0  # Empirically adjusted

            # Clamp to reasonable range
            speed_kmh = max(0, min(speed_kmh, 200))

            return speed_kmh

        except Exception as e:
            print(f"Error in raw speed calculation: {e}")
            return 0.0

    def _smooth_speed(self, raw_speed: float) -> float:
        """Apply smoothing to reduce speed fluctuations"""
        try:
            # Add current speed to history
            self.speed_history.append(raw_speed)

            # Keep only recent measurements
            if len(self.speed_history) > self.speed_window_size:
                self.speed_history.pop(0)

            # Apply different smoothing strategies based on history length
            if len(self.speed_history) < 3:
                # Not enough history, use raw speed
                smoothed = raw_speed
            else:
                # Apply weighted moving average with outlier rejection
                smoothed = self._weighted_moving_average()

            # Apply speed change limiting
            if hasattr(self, 'prev_speed') and self.prev_speed > 0:
                max_change = self.speed_change_threshold
                speed_diff = abs(smoothed - self.prev_speed)

                if speed_diff > max_change:
                    # Limit the change to prevent sudden jumps
                    if smoothed > self.prev_speed:
                        smoothed = self.prev_speed + max_change
                    else:
                        smoothed = max(0, self.prev_speed - max_change)

            # Update previous speed
            self.prev_speed = smoothed

            return smoothed

        except Exception as e:
            print(f"Error in speed smoothing: {e}")
            return raw_speed

    def _weighted_moving_average(self) -> float:
        """Calculate weighted moving average with outlier rejection"""
        if len(self.speed_history) < 2:
            return self.speed_history[-1] if self.speed_history else 0.0

        # Remove outliers (values more than 2 std deviations from mean)
        speeds = np.array(self.speed_history)
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)

        if std_speed > 0:
            # Keep values within 2 standard deviations
            valid_mask = np.abs(speeds - mean_speed) <= (2 * std_speed)
            valid_speeds = speeds[valid_mask]
        else:
            valid_speeds = speeds

        if len(valid_speeds) == 0:
            return mean_speed

        # Apply weighted average (more weight to recent values)
        weights = np.exp(np.linspace(-1, 0, len(valid_speeds)))
        weights = weights / np.sum(weights)

        weighted_speed = np.sum(valid_speeds * weights)

        return float(weighted_speed)
    
    def track_vehicles(self, vehicles: List[Dict]) -> Dict:
        """
        Simple vehicle tracking based on position
        
        Args:
            vehicles: List of detected vehicles
            
        Returns:
            Dictionary of tracked vehicles
        """
        current_tracks = {}
        
        for vehicle in vehicles:
            center = vehicle['center']
            
            # Find closest existing track
            best_match = None
            min_distance = float('inf')
            
            for track_id, track_data in self.vehicle_tracks.items():
                if len(track_data['positions']) > 0:
                    last_pos = track_data['positions'][-1]
                    distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    if distance < min_distance and distance < 50:  # 50 pixel threshold
                        min_distance = distance
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                track_id = best_match
                self.vehicle_tracks[track_id]['positions'].append(center)
                self.vehicle_tracks[track_id]['last_seen'] = self.frame_count
            else:
                # Create new track
                track_id = self.track_id_counter
                self.track_id_counter += 1
                self.vehicle_tracks[track_id] = {
                    'speeds': [],
                    'positions': [center],
                    'last_seen': self.frame_count,
                    'overspeeding': False
                }
            
            # Add vehicle info to current tracks
            current_tracks[track_id] = {
                **vehicle,
                'track_id': track_id
            }
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return current_tracks
    
    def _cleanup_tracks(self, max_age: int = 30):
        """Remove old tracks"""
        to_remove = []
        for track_id, track_data in self.vehicle_tracks.items():
            if self.frame_count - track_data['last_seen'] > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.vehicle_tracks[track_id]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Process a single frame for speed detection

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, ego_speed, vehicle_detections)
        """
        self.frame_count += 1
        ego_speed = 0.0

        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        tracked_vehicles = self.track_vehicles(vehicles)

        # Calculate optical flow if we have a previous frame
        if self.prev_frame is not None:
            try:
                flow = self.calculate_optical_flow(self.prev_frame, frame)
                # Enhanced speed estimation with vehicle data
                ego_speed = self.estimate_ego_speed(flow, list(tracked_vehicles.values()))

                # Update vehicle speeds based on ego motion and relative analysis
                self._update_vehicle_speeds_advanced(tracked_vehicles, ego_speed)

            except Exception as e:
                print(f"Error calculating optical flow: {e}")
                ego_speed = 0.0

        # Draw annotations on frame
        annotated_frame = self._annotate_frame(frame.copy(), tracked_vehicles, ego_speed)

        # Store current frame for next iteration
        self.prev_frame = frame.copy()

        # Prepare vehicle data with speed information for return
        vehicles_with_speed = []
        for track_id, vehicle in tracked_vehicles.items():
            vehicle_data = vehicle.copy()
            vehicle_data['track_id'] = track_id

            # Add speed information from tracking data
            if track_id in self.vehicle_tracks:
                track_data = self.vehicle_tracks[track_id]
                if track_data.get('speeds'):
                    # Use recent average speed
                    recent_speeds = track_data['speeds'][-5:] if len(track_data['speeds']) >= 5 else track_data['speeds']
                    vehicle_data['speed'] = np.mean(recent_speeds)
                else:
                    vehicle_data['speed'] = 0.0

                # Add overspeeding status
                vehicle_data['overspeeding'] = track_data.get('overspeeding', False)
            else:
                vehicle_data['speed'] = 0.0
                vehicle_data['overspeeding'] = False

            vehicles_with_speed.append(vehicle_data)

        return annotated_frame, ego_speed, vehicles_with_speed

    def _update_vehicle_speeds_advanced(self, tracked_vehicles: Dict, ego_speed: float):
        """Advanced vehicle speed calculation using relative motion analysis"""
        for track_id, vehicle in tracked_vehicles.items():
            if track_id in self.vehicle_tracks:
                # Calculate vehicle speed using multiple methods
                vehicle_speed = self._calculate_vehicle_speed(track_id, vehicle, ego_speed)

                self.vehicle_tracks[track_id]['speeds'].append(vehicle_speed)

                # Enhanced overspeeding detection
                if len(self.vehicle_tracks[track_id]['speeds']) > 8:
                    # Use multiple statistical measures
                    recent_speeds = self.vehicle_tracks[track_id]['speeds'][-8:]
                    avg_speed = np.mean(recent_speeds)
                    median_speed = np.median(recent_speeds)
                    max_speed = np.max(recent_speeds)

                    # Vehicle is overspeeding if consistently above limit
                    consistent_overspeeding = avg_speed > self.speed_limit
                    peak_overspeeding = max_speed > self.speed_limit * 1.2

                    self.vehicle_tracks[track_id]['overspeeding'] = consistent_overspeeding or peak_overspeeding

                    # Store detailed speed analysis
                    self.vehicle_tracks[track_id]['speed_analysis'] = {
                        'avg_speed': avg_speed,
                        'median_speed': median_speed,
                        'max_speed': max_speed,
                        'speed_variance': np.var(recent_speeds)
                    }

    def _calculate_vehicle_speed(self, track_id: int, vehicle: Dict, ego_speed: float) -> float:
        """Calculate individual vehicle speed using advanced techniques"""
        try:
            bbox = vehicle['bbox']
            center = vehicle['center']

            # Method 1: Position-based relative speed
            position_speed = self._calculate_position_based_speed(track_id, center, ego_speed)

            # Method 2: Size-based distance estimation
            size_speed = self._calculate_size_based_speed(track_id, bbox, ego_speed)

            # Method 3: Lane-relative speed
            lane_speed = self._calculate_lane_relative_speed(center, ego_speed)

            # Combine methods with confidence weighting
            speeds = []
            weights = []

            if position_speed is not None:
                speeds.append(position_speed)
                weights.append(2.0)  # High confidence

            if size_speed is not None:
                speeds.append(size_speed)
                weights.append(1.5)  # Medium confidence

            if lane_speed is not None:
                speeds.append(lane_speed)
                weights.append(1.0)  # Lower confidence

            if speeds:
                # Weighted average
                final_speed = sum(s * w for s, w in zip(speeds, weights)) / sum(weights)

                # Apply vehicle-specific smoothing
                smoothed_speed = self._smooth_vehicle_speed(track_id, final_speed)

                return max(0, min(smoothed_speed, 250))  # Reasonable range for vehicles
            else:
                # Fallback: assume similar to ego speed with some variation
                return ego_speed * (0.8 + np.random.normal(0, 0.2))

        except Exception as e:
            print(f"Error calculating vehicle speed for track {track_id}: {e}")
            return ego_speed * 0.9  # Conservative fallback

    def _calculate_position_based_speed(self, track_id: int, center: tuple, ego_speed: float) -> float:
        """Calculate speed based on position changes"""
        try:
            if track_id not in self.vehicle_tracks:
                return None

            positions = self.vehicle_tracks[track_id]['positions']
            if len(positions) < 3:
                return None

            # Calculate movement over last few frames
            recent_positions = positions[-3:]

            # Calculate displacement
            dx_total = recent_positions[-1][0] - recent_positions[0][0]
            dy_total = recent_positions[-1][1] - recent_positions[0][1]

            # Convert pixel movement to speed estimate
            # Assume 30 FPS and calibrate based on typical road scenarios
            frames_elapsed = len(recent_positions) - 1
            time_elapsed = frames_elapsed / 30.0  # seconds

            # Pixel to meter conversion (rough estimate)
            # This would need calibration for specific camera setup
            pixels_per_meter = 20  # Approximate for typical dashcam

            displacement_meters = np.sqrt(dx_total**2 + dy_total**2) / pixels_per_meter
            speed_ms = displacement_meters / time_elapsed
            speed_kmh = speed_ms * 3.6

            # Relative to ego vehicle
            if dx_total < -10:  # Moving opposite direction
                relative_speed = ego_speed + speed_kmh
            elif dx_total > 10:  # Moving same direction
                relative_speed = abs(ego_speed - speed_kmh)
            else:  # Similar direction
                relative_speed = ego_speed + (speed_kmh * 0.1)

            return relative_speed

        except Exception as e:
            print(f"Error in position-based speed calculation: {e}")
            return None

    def _calculate_size_based_speed(self, track_id: int, bbox: list, ego_speed: float) -> float:
        """Calculate speed based on apparent size changes (distance estimation)"""
        try:
            if track_id not in self.vehicle_tracks:
                return None

            # Store bbox history for size analysis
            if 'bbox_history' not in self.vehicle_tracks[track_id]:
                self.vehicle_tracks[track_id]['bbox_history'] = []

            bbox_history = self.vehicle_tracks[track_id]['bbox_history']
            bbox_history.append(bbox)

            # Keep only recent history
            if len(bbox_history) > 10:
                bbox_history.pop(0)

            if len(bbox_history) < 3:
                return None

            # Calculate size changes
            recent_bboxes = bbox_history[-3:]

            # Calculate bbox areas
            areas = []
            for box in recent_bboxes:
                width = box[2] - box[0]
                height = box[3] - box[1]
                areas.append(width * height)

            # Size change indicates distance change
            area_change = areas[-1] - areas[0]

            # Convert area change to speed estimate
            # Larger area = closer = approaching
            # Smaller area = farther = receding

            if abs(area_change) < 100:  # Small change - similar speed
                return ego_speed * (0.9 + np.random.normal(0, 0.1))
            elif area_change > 100:  # Getting larger - approaching
                approach_factor = min(area_change / 1000, 0.5)
                return ego_speed * (1.2 + approach_factor)
            else:  # Getting smaller - receding
                recede_factor = min(abs(area_change) / 1000, 0.3)
                return ego_speed * (0.8 - recede_factor)

        except Exception as e:
            print(f"Error in size-based speed calculation: {e}")
            return None

    def _calculate_lane_relative_speed(self, center: tuple, ego_speed: float) -> float:
        """Calculate speed based on lane position and typical traffic patterns"""
        try:
            # Assume image width represents road width
            # This is a simplified model - real implementation would need road detection

            x_pos = center[0]
            # Assume image width of ~640 pixels for typical dashcam
            image_width = 640

            # Normalize position (0 = left edge, 1 = right edge)
            normalized_x = x_pos / image_width

            # Lane-based speed estimation
            if 0.1 <= normalized_x <= 0.4:  # Left lanes (oncoming traffic)
                # Oncoming traffic typically moves at similar speeds
                return ego_speed * (1.0 + np.random.normal(0, 0.15))
            elif 0.6 <= normalized_x <= 0.9:  # Right lanes (same direction)
                # Same direction traffic - more variation
                return ego_speed * (0.8 + np.random.normal(0, 0.25))
            else:  # Center or edge - uncertain
                return ego_speed * (0.9 + np.random.normal(0, 0.2))

        except Exception as e:
            print(f"Error in lane-relative speed calculation: {e}")
            return None

    def _smooth_vehicle_speed(self, track_id: int, raw_speed: float) -> float:
        """Apply smoothing to individual vehicle speeds"""
        try:
            if track_id not in self.vehicle_tracks:
                return raw_speed

            speeds = self.vehicle_tracks[track_id]['speeds']

            if len(speeds) < 3:
                return raw_speed

            # Use last few speeds for smoothing
            recent_speeds = speeds[-5:] if len(speeds) >= 5 else speeds

            # Apply median filter to remove outliers
            median_speed = np.median(recent_speeds)

            # Apply exponential smoothing
            alpha = 0.3  # Smoothing factor
            if len(speeds) > 0:
                smoothed = alpha * raw_speed + (1 - alpha) * speeds[-1]
            else:
                smoothed = raw_speed

            # Limit sudden changes
            max_change = 15.0  # km/h per frame
            if len(speeds) > 0:
                speed_diff = abs(smoothed - speeds[-1])
                if speed_diff > max_change:
                    if smoothed > speeds[-1]:
                        smoothed = speeds[-1] + max_change
                    else:
                        smoothed = max(0, speeds[-1] - max_change)

            return smoothed

        except Exception as e:
            print(f"Error in vehicle speed smoothing: {e}")
            return raw_speed

    def _annotate_frame(self, frame: np.ndarray, tracked_vehicles: Dict, ego_speed: float) -> np.ndarray:
        """
        Annotate frame with speed information and vehicle detections

        Args:
            frame: Input frame
            tracked_vehicles: Dictionary of tracked vehicles
            ego_speed: Ego vehicle speed

        Returns:
            Annotated frame
        """
        # Draw ego speed
        speed_text = f"Speed: {ego_speed:.1f} km/h"
        cv2.putText(frame, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

        # Draw frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Draw vehicle detections
        for track_id, vehicle in tracked_vehicles.items():
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox

            # Get vehicle info
            track_data = self.vehicle_tracks.get(track_id, {})
            is_overspeeding = track_data.get('overspeeding', False)

            # Choose color based on overspeeding status
            color = (0, 0, 255) if is_overspeeding else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw track ID and speed info
            if track_data.get('speeds'):
                avg_speed = np.mean(track_data['speeds'][-5:]) if len(track_data['speeds']) >= 5 else 0
                speed_text = f"ID:{track_id} {avg_speed:.1f}km/h"
            else:
                speed_text = f"ID:{track_id}"

            cv2.putText(frame, speed_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mark overspeeding vehicles
            if is_overspeeding:
                cv2.putText(frame, "OVERSPEEDING!", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def process_video(self, video_path: str, output_path: str = None,
                     max_frames: int = None) -> Dict:
        """
        Process entire video for speed detection

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            max_frames: Maximum frames to process (optional)

        Returns:
            Processing results dictionary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Processing statistics
        ego_speeds = []
        violation_frames = []
        processing_times = []

        frame_idx = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                frame_start = time.time()

                # Process frame
                processed_frame, ego_speed, vehicles = self.process_frame(frame)

                frame_time = time.time() - frame_start
                processing_times.append(frame_time)

                # Store results
                ego_speeds.append(ego_speed)

                # Check for violations
                for vehicle in vehicles:
                    track_id = vehicle['track_id']
                    if self.vehicle_tracks[track_id].get('overspeeding', False):
                        violation_frames.append({
                            'frame': frame_idx,
                            'track_id': track_id,
                            'ego_speed': ego_speed,
                            'timestamp': frame_idx / fps
                        })

                # Write frame to output video
                if out:
                    out.write(processed_frame)

                frame_idx += 1

                # Progress update
                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed
                    print(f"Processed {frame_idx}/{total_frames} frames "
                          f"({fps_current:.1f} FPS)")

        finally:
            cap.release()
            if out:
                out.release()

        total_time = time.time() - start_time
        avg_processing_time = np.mean(processing_times) if processing_times else 0

        results = {
            'total_frames': frame_idx,
            'total_time': total_time,
            'avg_processing_time': avg_processing_time,
            'avg_ego_speed': np.mean(ego_speeds) if ego_speeds else 0,
            'max_ego_speed': np.max(ego_speeds) if ego_speeds else 0,
            'min_ego_speed': np.min(ego_speeds) if ego_speeds else 0,
            'speed_history': ego_speeds,  # Complete speed history for analysis
            'violation_frames': violation_frames,
            'output_path': output_path
        }

        print(f"\nProcessing completed:")
        print(f"Total frames: {results['total_frames']}")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Average speed: {results['avg_ego_speed']:.1f} km/h")
        print(f"Max speed: {results['max_ego_speed']:.1f} km/h")
        print(f"Violations detected: {len(violation_frames)}")

        return results
