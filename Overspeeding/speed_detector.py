import cv2
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
import torch

class SpeedDetector:
    def __init__(self, video_path, output_dir='outputs'):
        self.cap = cv2.VideoCapture(video_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Constants
        self.RECORDING_SPEED = 100  # km/h (speed of recording vehicle)
        self.SPEED_LIMIT = 120  # km/h
        
        # ROI for road area (bottom 60% of frame)
        self.ROI = (
            int(self.frame_height * 0.4),  # y1 (40% from top)
            self.frame_height,              # y2 (bottom)
            0,                              # x1 (left)
            self.frame_width                # x2 (right)
        )
        
        # Speed estimation parameters
        self.SPEED_SCALE = 0.02  # pixels/(km/h) - calibration factor
        self.FRAME_SKIP = 2      # Process every nth frame for performance
        
        # Initialize YOLOv5 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_yolo_model()
        
        # Vehicle classes we're interested in
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Initialize previous frame storage
        self.prev_processed = None
        
        # Vehicle tracking with speed history
        self.vehicles = defaultdict(lambda: {
            'speeds': [],
            'bbox': None,
            'overspeeding': False
        })
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(output_dir, 'output.avi')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                 (self.frame_width, self.frame_height))

    def load_yolo_model(self):
        """Load YOLOv5 model"""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(self.device)
        model.conf = 0.5
        model.iou = 0.45
        return model

    def preprocess_frame(self, frame):
        """Preprocess frame for optical flow calculation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 1.5)

    def create_roi_mask(self):
        """Create mask for region of interest"""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        y1, y2, x1, x2 = self.ROI
        mask[y1:y2, x1:x2] = 1
        return mask

    def estimate_speed_from_flow(self, flow, vehicle_bbox=None):
        """Estimate speed from optical flow"""
        if vehicle_bbox is None:
            # Use entire ROI if no specific vehicle bbox
            y1, y2, x1, x2 = self.ROI
            flow_roi = flow[y1:y2, x1:x2]
            mean_flow = np.mean(np.abs(flow_roi))
        else:
            # Use vehicle bbox area
            x1, y1, x2, y2 = vehicle_bbox
            flow_roi = flow[y1:y2, x1:x2]
            mean_flow = np.mean(np.abs(flow_roi))

        # Convert flow to speed
        relative_speed = mean_flow / self.SPEED_SCALE
        
        # Add recording vehicle speed if moving in same direction
        if relative_speed < self.RECORDING_SPEED:
            return self.RECORDING_SPEED + relative_speed
        return relative_speed

    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        
        vehicles = []
        detections = results.xyxy[0].cpu().numpy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) in self.vehicle_classes and conf > 0.5:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # Only consider vehicles in ROI
                if y2 > self.ROI[0]:  # If vehicle bottom is in ROI
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': int(cls)
                    })
        
        return vehicles

    def process_frame(self, frame, frame_number):
        """Process a single frame"""
        # Skip frames if needed
        if frame_number % self.FRAME_SKIP != 0:
            return frame

        # Preprocess frame
        processed = self.preprocess_frame(frame)
        
        # Calculate optical flow if we have a previous frame
        if self.prev_processed is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_processed, processed,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Detect vehicles
            vehicles = self.detect_vehicles(frame)
            
            # Process each detected vehicle
            for vehicle in vehicles:
                bbox = vehicle['bbox']
                vehicle_class = vehicle['class']
                
                # Generate vehicle ID based on position
                vehicle_id = f"vehicle_{bbox[0]}_{bbox[1]}"
                
                # Estimate vehicle speed using flow in bbox area
                speed = self.estimate_speed_from_flow(flow, bbox)
                
                # Update vehicle speed history
                self.vehicles[vehicle_id]['speeds'].append(speed)
                self.vehicles[vehicle_id]['bbox'] = bbox
                
                # Calculate median speed from history
                if len(self.vehicles[vehicle_id]['speeds']) > 5:
                    median_speed = np.median(self.vehicles[vehicle_id]['speeds'][-5:])
                else:
                    median_speed = speed
                
                # Draw vehicle information
                color = (0, 255, 0)  # Default green
                if median_speed > self.SPEED_LIMIT:
                    color = (0, 0, 255)  # Red for overspeeding
                    if not self.vehicles[vehicle_id]['overspeeding']:
                        self.vehicles[vehicle_id]['overspeeding'] = True
                        # Save violation image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        violation_path = os.path.join(
                            self.output_dir,
                            f"violation_{vehicle_id}_{median_speed:.1f}kmh_{timestamp}.jpg"
                        )
                        cv2.imwrite(violation_path, frame)
                
                # Draw bounding box and speed
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{median_speed:.1f} km/h", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw ROI
            y1, y2, x1, x2 = self.ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Update previous frame
        self.prev_processed = processed.copy()
        
        # Add speed limit and recording speed overlay
        cv2.putText(frame, f"Speed Limit: {self.SPEED_LIMIT} km/h", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Recording Speed: {self.RECORDING_SPEED} km/h", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def process_video(self):
        """Process the entire video"""
        frame_number = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame, frame_number)
            
            # Write frame
            self.out.write(processed_frame)
            
            # Display progress
            if frame_number % 30 == 0:
                print(f"Processing frame {frame_number}")
            
            frame_number += 1
        
        # Cleanup
        self.cap.release()
        self.out.release()
        print("Video processing complete!")
