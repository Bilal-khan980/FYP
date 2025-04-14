import os
import sys
import cv2
import numpy as np
import torch
import uuid
import shutil
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime

# Create FastAPI app
app = FastAPI(title="Overspeeding Detection API", description="API for processing videos to detect overspeeding vehicles")

# Create outputs directory
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Constants
SPEED_LIMIT = 110  # km/h (constant speed as mentioned in requirements)
# Line positions: 1st line at 30% from bottom (reference line), 2nd line at 40% from bottom
LINE_POSITIONS = [0.75, 0.65]  # Relative positions of the two horizontal lines (bottom to top)
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLO detection
IOU_THRESHOLD = 0.45  # IOU threshold for YOLO detection
MAX_TRACKING_DISTANCE = 50  # Maximum distance in pixels to associate the same vehicle between frames

class SpeedDetector:
    def __init__(self, video_path="data/input.mp4", output_dir=outputs_dir):
        """
        Initialize the speed detector

        Args:
            video_path (str): Path to the input video (default: data/input.mp4)
            output_dir (str): Directory to save output images
        """
        self.video_path = video_path
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load YOLOv5 model
        self.model = self.load_yolo_model()

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Calculate line positions in pixels
        self.line_positions = [int(pos * self.frame_height) for pos in LINE_POSITIONS]

        # Print line positions for debugging
        print("\nLine positions (from bottom of frame):")
        for i, pos in enumerate(LINE_POSITIONS):
            pixel_pos = self.line_positions[i]
            percent_from_top = int(pixel_pos / self.frame_height * 100)
            percent_from_bottom = int((self.frame_height - pixel_pos) / self.frame_height * 100)
            print(f"Line {i+1}: {pos:.2f} relative position, {pixel_pos} pixels from top, {self.frame_height - pixel_pos} pixels from bottom")
            print(f"       {percent_from_top}% from top, {percent_from_bottom}% from bottom")

        # Dictionary to track vehicles
        # {id: {line1_time, line2_time, captured, position, last_seen, current_line, reference_frame}}
        self.vehicles = defaultdict(lambda: {'captured': False, 'line_states': [False, False], 'current_line': 0, 'reference_frame': None})

        # Vehicle ID counter
        self.next_vehicle_id = 0

        # Previous frame's vehicles for tracking
        self.prev_vehicles = []

        # Frame counter
        self.frame_count = 0

        # Create output video writer
        self.output_video_path = os.path.join(output_dir, 'output_video.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                  (self.frame_width, self.frame_height))

    def load_yolo_model(self):
        """Load YOLOv5 model"""
        # Load model from local path
        model_path = os.path.join('modelss', 'yolov5s.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv5 model not found at: {model_path}")

        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.conf = CONFIDENCE_THRESHOLD
        model.iou = IOU_THRESHOLD
        model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        return model

    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame using YOLOv5

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            list: List of detected vehicles with bounding boxes and centers
        """
        # Run inference
        results = self.model(frame)

        # Process results
        vehicles = []
        for *box, conf, _ in results.xyxy[0].cpu().numpy():  # Using _ to ignore cls
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': float(conf)
            })

        return vehicles

    def track_vehicles(self, current_vehicles, frame_time, frame):
        """
        Track vehicles and check if they cross the lines

        Args:
            current_vehicles (list): List of detected vehicles in current frame
            frame_time (float): Current frame timestamp
            frame (numpy.ndarray): Current frame for visualization
        """
        # Match current vehicles with previous frame's vehicles
        matched_vehicle_ids = []

        # For each vehicle in the current frame
        for vehicle in current_vehicles:
            center_x, center_y = vehicle['center']
            vehicle_id = None
            min_distance = float('inf')

            # Try to match with a vehicle from the previous frame
            for prev_vehicle in self.prev_vehicles:
                prev_center_x, prev_center_y = prev_vehicle['center']
                prev_id = prev_vehicle['id']

                # Calculate distance between centers
                distance = np.sqrt((center_x - prev_center_x)*2 + (center_y - prev_center_y)**2)

                # If the distance is small enough and this ID hasn't been matched yet
                if distance < MAX_TRACKING_DISTANCE and distance < min_distance and prev_id not in matched_vehicle_ids:
                    min_distance = distance
                    vehicle_id = prev_id

            # If no match found, assign a new ID
            if vehicle_id is None:
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1

            # Add ID to the vehicle and to matched IDs
            vehicle['id'] = vehicle_id
            matched_vehicle_ids.append(vehicle_id)

            # Update vehicle data
            self.vehicles[vehicle_id]['position'] = (center_x, center_y)
            self.vehicles[vehicle_id]['last_seen'] = self.frame_count
            self.vehicles[vehicle_id]['bbox'] = vehicle['bbox']

            # Check if the vehicle crosses any of the lines
            self.check_line_crossing(vehicle_id, center_y, frame_time, frame)

        # Update previous vehicles for next frame
        self.prev_vehicles = []
        for vehicle in current_vehicles:
            self.prev_vehicles.append(vehicle.copy())

        # Remove vehicles that haven't been seen for a while
        vehicles_to_remove = []
        for vehicle_id in self.vehicles:
            if self.frame_count - self.vehicles[vehicle_id].get('last_seen', 0) > 30:  # 30 frames ~ 1 second at 30 fps
                vehicles_to_remove.append(vehicle_id)

        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]

    def check_line_crossing(self, vehicle_id, center_y, frame_time, frame):
        """
        Check if a vehicle crosses any of the horizontal lines from bottom to top

        Args:
            vehicle_id (int): Vehicle ID
            center_y (int): Y-coordinate of the vehicle center
            frame_time (float): Current frame timestamp
            frame (numpy.ndarray): Current frame for visualization
        """
        # Get the vehicle's direction if available
        if 'direction' not in self.vehicles[vehicle_id]:
            # Initialize direction as None (unknown)
            self.vehicles[vehicle_id]['direction'] = None

        # Store previous position if available
        prev_y = self.vehicles[vehicle_id].get('prev_y', None)

        # If we have a previous position, determine direction
        if prev_y is not None:
            # In image coordinates, y decreases as you move up
            if center_y < prev_y:
                # Moving upward (bottom to top)
                self.vehicles[vehicle_id]['direction'] = 'up'
            elif center_y > prev_y:
                # Moving downward (top to bottom)
                self.vehicles[vehicle_id]['direction'] = 'down'

        # Store current y position for next frame
        self.vehicles[vehicle_id]['prev_y'] = center_y

        # Only process line crossings if vehicle is moving upward
        if self.vehicles[vehicle_id]['direction'] == 'up':
            # Check each line
            for i, line_y in enumerate(self.line_positions):
                # If the center is close to the line (within 15 pixels for even better detection)
                if abs(center_y - line_y) < 15:
                    # Record line crossing
                    line_idx = i  # 0 or 1 for the two lines

                    # Update line crossing state
                    prev_state = self.vehicles[vehicle_id]['line_states'][line_idx]
                    self.vehicles[vehicle_id]['line_states'][line_idx] = True

                    # If this is a new crossing (state changed from False to True)
                    if not prev_state:
                        # Record the time when the vehicle crosses the line
                        self.vehicles[vehicle_id][f'line{line_idx+1}_time'] = frame_time

                        # Add text to the frame
                        x, y = self.vehicles[vehicle_id]['position']
                        cv2.putText(frame, f"Line {line_idx+1} crossed (bottom to top)", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # If this is the first line (reference line), capture the frame
                        if line_idx == 0:  # First line (reference line)
                            # Store a copy of the current frame for this vehicle
                            self.vehicles[vehicle_id]['reference_frame'] = frame.copy()
                            # Update the vehicle's current line status
                            self.vehicles[vehicle_id]['current_line'] = 1
                        # If vehicle has already crossed line 1, update to line 2
                        elif self.vehicles[vehicle_id]['line_states'][0]:  # If first line was crossed
                            # Update the vehicle's current line status
                            self.vehicles[vehicle_id]['current_line'] = line_idx + 1
                            # Check for overspeeding if this is line 2
                            if line_idx == 1:  # Second line (top)
                                print(f"Vehicle {vehicle_id} crossed line 2 - directly capturing screenshot")
                                # Force capture a screenshot when crossing line 2
                                self.vehicles[vehicle_id]['captured'] = True
                                self.vehicles[vehicle_id]['current_line'] = 2
                                # Directly capture screenshot without additional checks
                                self.capture_overspeeding_vehicle(vehicle_id, frame)
                                # Also run the regular check
                                self.check_overspeeding(vehicle_id, frame)

    def check_overspeeding(self, vehicle_id, frame):
        """
        Check if a vehicle has crossed both lines and label it as overspeeding
        Capture a screenshot from the original video

        Args:
            vehicle_id (int): Vehicle ID
            frame (numpy.ndarray): Current frame
        """
        vehicle_data = self.vehicles[vehicle_id]

        # Check if the vehicle has crossed both lines
        print(f"Checking vehicle {vehicle_id} for overspeeding")
        if all(f'line{i+1}_time' in vehicle_data for i in range(2)):
            print(f"Vehicle {vehicle_id} has crossed both lines")
            # SIMPLIFIED: Always mark vehicles that cross both lines as overspeeding
            # This ensures we capture all vehicles that cross both lines
            if vehicle_data['line1_time'] < vehicle_data['line2_time']:
                print(f"Vehicle {vehicle_id} crossed lines in correct order - MARKING AS OVERSPEEDING")

                # Label the vehicle as overspeeding
                vehicle_data['captured'] = True

                # Get vehicle position and bounding box
                x, y = vehicle_data['position']
                x1, y1, x2, y2 = vehicle_data['bbox']

                # Add text to the frame
                cv2.putText(frame, "OVERSPEEDING", (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw a thick red bounding box around the vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Thicker red box

                # Capture a screenshot from the reference frame
                if vehicle_data['reference_frame'] is not None:
                    print(f"Vehicle {vehicle_id} crossed both lines - capturing screenshot")
                    self.capture_overspeeding_vehicle(vehicle_id, vehicle_data['reference_frame'])
                else:
                    # Fallback to current frame if reference frame wasn't captured
                    print(f"Vehicle {vehicle_id} crossed both lines - capturing screenshot (fallback to current frame)")
                    self.capture_overspeeding_vehicle(vehicle_id, frame)

            # Check if the vehicle slowed down (crossed line2 then went back to line1)
            elif ('line1_time' in vehicle_data and 'line2_time' in vehicle_data and
                  vehicle_data['line2_time'] < vehicle_data.get('line1_time', float('inf'))):
                # Vehicle slowed down, disregard it
                x, y = vehicle_data['position']
                cv2.putText(frame, "SLOWED DOWN", (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def capture_vehicle(self, vehicle_id, frame, speed):
        """
        Capture an image of a speeding vehicle

        Args:
            vehicle_id (int): Vehicle ID
            frame (numpy.ndarray): Current frame
            speed (float): Calculated speed
        """
        # Extract the vehicle from the frame
        x1, y1, x2, y2 = self.vehicles[vehicle_id]['bbox']
        vehicle_img = frame[max(0, y1-50):min(self.frame_height, y2+50),
                           max(0, x1-50):min(self.frame_width, x2+50)]

        # Save the image
        output_path = os.path.join(self.output_dir, f"speeding_vehicle_{vehicle_id}_{speed:.1f}kmh.jpg")
        cv2.imwrite(output_path, vehicle_img)
        print(f"Captured speeding vehicle: {output_path}")

    def capture_overspeeding_vehicle(self, vehicle_id, original_frame):
        """
        Capture an image of a vehicle labeled as OVERSPEEDING using the original frame

        Args:
            vehicle_id (int): Vehicle ID
            original_frame (numpy.ndarray): Original frame without annotations
        """
        try:
            print(f"Starting to capture overspeeding vehicle {vehicle_id}")
            # Extract the vehicle from the original frame
            x1, y1, x2, y2 = self.vehicles[vehicle_id]['bbox']

            # Create a copy of the frame to add annotations
            annotated_frame = original_frame.copy()

            # Draw a red bounding box around the vehicle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Thick red box

            # Add OVERSPEEDING label
            cv2.putText(annotated_frame, "OVERSPEEDING", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add some padding around the bounding box for the cropped image
            padding = 50
            vehicle_img = original_frame[max(0, y1-padding):min(self.frame_height, y2+padding),
                                       max(0, x1-padding):min(self.frame_width, x2+padding)]

            # Save both the full annotated frame and the cropped vehicle image
            timestamp = self.frame_count  # Use frame count as timestamp

            # Create unique filenames for each vehicle and frame
            full_output_path = os.path.join(self.output_dir, f"overspeeding_vehicle_{vehicle_id}_full_frame{timestamp}.jpg")
            crop_output_path = os.path.join(self.output_dir, f"overspeeding_vehicle_{vehicle_id}_cropped_frame{timestamp}.jpg")

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save the images
            success1 = cv2.imwrite(full_output_path, annotated_frame)
            success2 = cv2.imwrite(crop_output_path, vehicle_img)

            if success1 and success2:
                print(f"Successfully captured overspeeding vehicle from original frame:")
                print(f"  - Full frame: {full_output_path}")
                print(f"  - Cropped: {crop_output_path}")
            else:
                print(f"WARNING: Failed to save one or more images for vehicle {vehicle_id}")
        except Exception as e:
            print(f"ERROR capturing overspeeding vehicle {vehicle_id}: {str(e)}")

    def process_video(self, max_frames=None, timeout=60):
        """Process the input video and detect speeding vehicles

        Args:
            max_frames (int, optional): Maximum number of frames to process. Defaults to None (process all frames).
            timeout (int, optional): Maximum processing time in seconds. Defaults to 60.
        """
        import time

        self.frame_count = 0
        start_time = time.time()

        # Get total frame count from the video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in input video: {total_frames}")

        # If max_frames is not specified, process all frames
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)

        while self.frame_count < max_frames:
            # Check if timeout has been reached
            if time.time() - start_time > timeout:
                print(f"\nTimeout reached after {timeout} seconds. Stopping processing.")
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_time = self.frame_count / self.fps

            # Print progress every 100 frames
            if self.frame_count % 100 == 0:
                print(f"Processing frame {self.frame_count}/{total_frames} ({self.frame_count/total_frames*100:.1f}%)")

            # Create a copy of the frame for drawing
            display_frame = frame.copy()

            # Draw the two horizontal lines (Line 1 at bottom as reference, Line 2 at top)
            for i, y in enumerate(self.line_positions):
                # Use different colors for each line to make them more visible
                line_colors = [(0, 165, 255), (0, 0, 255)]  # Orange (reference), Red
                cv2.line(display_frame, (0, y), (self.frame_width, y), line_colors[i], 3)  # Thicker line

                # Add special label for reference line
                if i == 0:
                    cv2.putText(display_frame, f"Line {i+1} - REFERENCE ({y} px, {int(y/self.frame_height*100)}% from top)", (10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_colors[i], 2)
                else:
                    cv2.putText(display_frame, f"Line {i+1} ({y} px, {int(y/self.frame_height*100)}% from top)", (10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_colors[i], 2)

            # Add speed limit text
            cv2.putText(display_frame, f"Speed Limit: {SPEED_LIMIT} km/h", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Detect vehicles
            vehicles = self.detect_vehicles(frame)

            # Track vehicles and check for speeding
            self.track_vehicles(vehicles, frame_time, display_frame)

            # Draw vehicle centers, bounding boxes, and line status
            for vehicle in vehicles:
                if 'id' in vehicle:  # Only process vehicles that have been assigned an ID
                    vehicle_id = vehicle['id']
                    center_x, center_y = vehicle['center']
                    x1, y1, x2, y2 = vehicle['bbox']

                    # Get the current line status
                    current_line = self.vehicles[vehicle_id].get('current_line', 0)

                    # Only label vehicles that have crossed the first line
                    if self.vehicles[vehicle_id]['line_states'][0]:  # If first line was crossed
                        # Set color based on line status
                        if current_line == 2 or self.vehicles[vehicle_id].get('captured', False):
                            # Red for overspeeding (crossed line 2)
                            box_color = (0, 0, 255)  # Red in BGR
                            label = "OVERSPEEDING"

                            # Draw a thick red bounding box around the vehicle
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)  # Thicker red box

                            # Capture a screenshot of the overspeeding vehicle if not already captured
                            if not self.vehicles[vehicle_id].get('screenshot_captured', False):
                                print(f"Capturing screenshot for vehicle {vehicle_id} in draw_vehicles")
                                self.capture_overspeeding_vehicle(vehicle_id, frame)  # Use original frame instead of display_frame
                                self.vehicles[vehicle_id]['screenshot_captured'] = True
                        elif current_line == 1:
                            # Green for line 1
                            box_color = (0, 255, 0)  # Green in BGR
                            label = "1"
                        else:
                            # Should not happen, but just in case
                            box_color = (255, 0, 0)  # Blue in BGR
                            label = ""

                        # Draw bounding box with color based on line status
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                        # Draw label if there is one
                        if label:
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    else:
                        # Draw a neutral bounding box for vehicles that haven't crossed any line
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Thin blue box

                    # Draw center dot for all vehicles
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Add frame counter
            cv2.putText(display_frame, f"Frame: {self.frame_count}/{total_frames}", (self.frame_width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Write the frame to output video (no display)
            self.out.write(display_frame)

        # Release resources
        self.cap.release()
        self.out.release()

        print(f"Output video saved to: {self.output_video_path}")
        print(f"Total frames processed: {self.frame_count}")

def process_video_api(video_path, cleanup=True):
    """Process video for overspeeding detection

    Args:
        video_path: Path to the input video
        cleanup: Whether to remove the input video after processing

    Returns:
        Tuple containing (result_message, image_path) where:
        - result_message: "Overspeeding violation detected" or "No violation"
        - image_path: Path to the saved image if violation detected, None otherwise
    """
    try:
        # Create a SpeedDetector instance
        detector = SpeedDetector(video_path=video_path)

        # Process the video with a timeout to prevent hanging
        detector.process_video(timeout=300)  # 5 minutes timeout

        # Check if any overspeeding vehicles were detected
        violation_detected = False
        violation_image_path = None

        # Look for the most recent overspeeding vehicle image
        for filename in os.listdir(detector.output_dir):
            if filename.startswith("overspeeding_vehicle_") and filename.endswith(".jpg") and "full_frame" in filename:
                violation_detected = True
                # Use the full frame image as the violation proof
                violation_image_path = os.path.join(detector.output_dir, filename)
                # Copy the image to the outputs directory with a standardized name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"violation_{timestamp}_overspeeding.jpg"
                new_path = os.path.join(outputs_dir, new_filename)
                shutil.copy(violation_image_path, new_path)
                violation_image_path = new_path
                break

        # Clean up the input file if requested
        if cleanup and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed temporary file: {video_path}")
            except Exception as e:
                print(f"Failed to remove temporary file {video_path}: {e}")

        if violation_detected:
            return "Overspeeding violation detected", violation_image_path
        else:
            return "No violation", None
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return f"Error processing video: {str(e)}", None

# FastAPI endpoints
@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Process a video file to detect overspeeding vehicles.

    Args:
        file: The video file to process

    Returns:
        JSON response with the result and image path if violation detected
    """
    # Generate a unique filename for the uploaded video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"

    # Save the uploaded file
    temp_file_path = os.path.join(current_dir, filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video synchronously - don't clean up the file so it can be viewed in the frontend
    result_message, image_path = process_video_api(
        temp_file_path,
        cleanup=False
    )

    # Prepare the response
    response_data = {
        "message": result_message,
    }

    if image_path:
        # Get just the filename from the path
        image_filename = os.path.basename(image_path)
        response_data["violation_detected"] = True
        response_data["image_url"] = f"/images/{image_filename}"
    else:
        response_data["violation_detected"] = False

    return JSONResponse(content=response_data)

@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Get a violation image file.

    Args:
        filename: The name of the image file

    Returns:
        The image file
    """
    file_path = os.path.join(outputs_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/jpeg"
    )

@app.get("/")
async def root():
    return {"message": "Overspeeding Detection API is running"}

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Detect speeding vehicles in a video')
    parser.add_argument('video_path', type=str, help='Path to the input video')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images')

    args = parser.parse_args()

    detector = SpeedDetector(args.video_path, args.output)
    detector.process_video()

if __name__ == '__main__':
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8005)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
        main()