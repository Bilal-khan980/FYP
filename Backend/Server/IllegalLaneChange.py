import os
import cv2
import torch
import numpy as np  # Used for array operations with contours
import datetime
import uuid
import shutil
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

print("Start !!!")
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create FastAPI app
app = FastAPI(title="Illegal Lane Change Detection API", description="API for processing videos to detect vehicles making illegal lane changes")

# Create outputs directory
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Add this new class to track vehicles
class VehicleTracker:
    def __init__(self, violation_threshold_frames=12):
        self.vehicles = {}
        self.violation_threshold_frames = violation_threshold_frames
        self.cleanup_threshold = 20

    def update_vehicle(self, vehicle_box, frame_number, is_violating):
        x1, y1, x2, y2 = vehicle_box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Find closest vehicle from previous frames
        closest_id = None
        min_distance = 100

        for vehicle_id, data in self.vehicles.items():
            if data['last_position'] is None:
                continue

            prev_center = data['last_position']
            distance = ((center[0] - prev_center[0]) ** 2 + 
                       (center[1] - prev_center[1]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_id = vehicle_id

        # Create new vehicle if no match found
        if closest_id is None:
            closest_id = len(self.vehicles)
            self.vehicles[closest_id] = {
                'last_position': None,
                'last_seen': frame_number,
                'is_violating': False,
                'violation_frames': 0,
                'violation_detected': False,
                'sustained_violation': False,  # Add this field
                'consecutive_violation_frames': 0  # Add this field
            }

        # Update vehicle data
        vehicle = self.vehicles[closest_id]
        vehicle['last_position'] = center
        vehicle['last_seen'] = frame_number

        if is_violating:
            vehicle['is_violating'] = True
            vehicle['violation_frames'] += 1
            vehicle['consecutive_violation_frames'] += 1
            if vehicle['consecutive_violation_frames'] >= self.violation_threshold_frames:
                vehicle['sustained_violation'] = True
                vehicle['violation_detected'] = True
                return True, closest_id, vehicle
        else:
            vehicle['is_violating'] = False
            vehicle['violation_frames'] = 0
            vehicle['consecutive_violation_frames'] = 0
            vehicle['sustained_violation'] = False

        return False, closest_id, vehicle

    def cleanup(self, current_frame):
        to_remove = []
        for vehicle_id, data in self.vehicles.items():
            if current_frame - data['last_seen'] > self.cleanup_threshold:
                to_remove.append(vehicle_id)

        for vehicle_id in to_remove:
            del self.vehicles[vehicle_id]

# Load the lane detection model
def load_lane_model():
    model_path = os.path.join('modelss', 'laneDetecion.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lane detection model not found at: {model_path}")

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.to(device)
    return model

# Load the car detection model
def load_car_model():
    model_path = os.path.join('modelss', 'yolov5s.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Car detection model not found at: {model_path}")

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.to(device)
    return model

# Detect cars in a frame and draw green bounding boxes
def detect_cars(frame, car_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = car_model(frame_rgb)

    output_frame = frame.copy()
    detections = results.xyxy[0].cpu().numpy()
    car_boxes = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls in [2, 5, 7] and conf > 0.5:  # Only cars, buses, trucks with confidence > 0.5
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            car_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Vehicle: {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_frame, car_boxes

def connect_lane_boxes(lane_boxes, frame_shape, min_distance=100):
    """Connect lane boxes to form continuous lanes, ensuring bottom connections.

    Args:
        lane_boxes: List of (x1, y1, x2, y2) coordinates for detected lane markings
        frame_shape: Tuple of (height, width) of the frame
        min_distance: Minimum distance to consider two boxes as part of same lane

    Returns:
        left_lane_points: List of points forming the left lane
        right_lane_points: List of points forming the right lane
    """
    if not lane_boxes:
        return [], []

    # Sort boxes by y-coordinate (bottom to top)
    lane_boxes = sorted(lane_boxes, key=lambda box: -(box[3] + box[1])/2)

    # Initialize lanes
    left_lane_points = []
    right_lane_points = []

    # Find lane centers
    centers = []
    for box in lane_boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))

    frame_center_x = frame_shape[1] / 2
    frame_height = frame_shape[0]

    # Separate into left and right lanes based on position
    left_groups = []
    right_groups = []

    # First pass: group points as before
    for x, y in centers:
        if x < frame_center_x:  # Left side
            added = False
            for group in left_groups:
                last_x, last_y = group[-1]
                if abs(x - last_x) < min_distance and abs(y - last_y) < min_distance:
                    group.append((x, y))
                    added = True
                    break
            if not added:
                left_groups.append([(x, y)])
        else:  # Right side
            added = False
            for group in right_groups:
                last_x, last_y = group[-1]
                if abs(x - last_x) < min_distance and abs(y - last_y) < min_distance:
                    group.append((x, y))
                    added = True
                    break
            if not added:
                right_groups.append([(x, y)])

    # Select longest groups
    if left_groups:
        left_lane_points = max(left_groups, key=len)
    if right_groups:
        right_lane_points = max(right_groups, key=len)

    # Add bottom points if needed
    bottom_y = frame_height - 20  # 20 pixels from bottom

    if left_lane_points:
        # Extrapolate bottom point for left lane
        if left_lane_points[-1][1] < bottom_y:
            last_points = left_lane_points[-2:]
            if len(last_points) >= 2:
                x1, y1 = last_points[0]
                x2, y2 = last_points[1]
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    bottom_x = x2 + slope * (bottom_y - y2)
                    left_lane_points.append((bottom_x, bottom_y))

    if right_lane_points:
        # Extrapolate bottom point for right lane
        if right_lane_points[-1][1] < bottom_y:
            last_points = right_lane_points[-2:]
            if len(last_points) >= 2:
                x1, y1 = last_points[0]
                x2, y2 = last_points[1]
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    bottom_x = x2 + slope * (bottom_y - y2)
                    right_lane_points.append((bottom_x, bottom_y))

    # Interpolate points to make lanes smoother
    if len(left_lane_points) > 1:
        left_lane_points = interpolate_points(left_lane_points, num_points=30)  # Increased points for smoother curve
    if len(right_lane_points) > 1:
        right_lane_points = interpolate_points(right_lane_points, num_points=30)  # Increased points for smoother curve

    return left_lane_points, right_lane_points

def interpolate_points(points, num_points=20):
    """Interpolate between points to create a smoother line."""
    if len(points) < 2:
        return points

    x_coords, y_coords = zip(*points)

    # Create parameter for interpolation (cumulative distance along points)
    t = np.zeros(len(points))
    for i in range(1, len(points)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)

    # Normalize parameter
    t = t / t[-1]

    # Interpolate
    t_new = np.linspace(0, 1, num_points)
    x_new = np.interp(t_new, t, x_coords)
    y_new = np.interp(t_new, t, y_coords)

    return list(zip(x_new, y_new))

def is_car_on_lane(car_box, lane_box, frame=None):
    """Check if car bounding box intersects with lane bounding box.
    
    Args:
        car_box: Tuple of (x1, y1, x2, y2) coordinates of the car
        lane_box: Tuple of (x1, y1, x2, y2) coordinates of the lane
        frame: Original frame (optional)
        
    Returns:
        bool: True if boxes intersect
        float: Overlap percentage (always 100 if intersecting)
        frame: Unmodified frame
    """
    x1_car, y1_car, x2_car, y2_car = car_box
    x1_lane, y1_lane, x2_lane, y2_lane = lane_box
    
    # Check if boxes intersect
    if (x2_car < x1_lane or x1_car > x2_lane or 
        y2_car < y1_lane or y1_car > y2_lane):
        return False, 0, frame
        
    return True, 100, frame

# Process a single frame to detect lane lines and cars
def process_frame(frame, lane_model, car_model, frame_number, vehicle_tracker=None):
    """Process a single frame for lane violation detection."""
    # Detect cars
    _, car_boxes = detect_cars(frame, car_model)
    
    # Detect lanes
    results = lane_model(frame)
    lane_boxes = []
    
    # Get solid lane boxes from detection results
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            lane_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    violations_info = []
    
    # Check each car against each lane
    for car_box in car_boxes:
        car_is_violating = False
        for lane_box in lane_boxes:
            is_violating, _, _ = is_car_on_lane(car_box, lane_box, frame)
            if is_violating:
                car_is_violating = True
                break
                
        if vehicle_tracker is not None and car_is_violating:
            sustained_violation, vehicle_id, vehicle_data = vehicle_tracker.update_vehicle(
                car_box, frame_number, car_is_violating
            )
            
            if vehicle_data['sustained_violation']:
                violations_info.append({
                    'vehicle_id': vehicle_id,
                    'sustained': True,
                    'consecutive_frames': vehicle_data['consecutive_violation_frames']
                })
                return frame, True, violations_info, True
    
    # Cleanup tracker
    if vehicle_tracker is not None:
        vehicle_tracker.cleanup(frame_number)
    
    return frame, len(violations_info) > 0, violations_info, False

def is_violation(car_box, lane_box):
    """Detect if a car is violating a lane by checking if 60% of its bottom center area intersects with the solid lane

    Args:
        car_box: Tuple of (x1, y1, x2, y2) coordinates of the car bounding box
        lane_box: Tuple of (x1, y1, x2, y2) coordinates of the lane bounding box

    Returns:
        bool: True if violation detected, False otherwise
        float: Percentage of overlap (0-100)
    """
    x1_car, y1_car, x2_car, y2_car = map(int, car_box)
    x1_lane, y1_lane, x2_lane, y2_lane = map(int, lane_box)

    # Calculate car dimensions
    car_width = x2_car - x1_car
    car_height = y2_car - y1_car

    # If car dimensions are invalid, return False
    if car_width <= 0 or car_height <= 0:
        return False, 0.0

    # Define the bottom center 60% area of the car
    center_x = (x1_car + x2_car) / 2
    bottom_center_width = car_width * 0.6  # 60% of car width
    bottom_center_left = center_x - bottom_center_width / 2
    bottom_center_right = center_x + bottom_center_width / 2

    # Check if the bottom center area intersects with the lane box
    # Calculate intersection of the bottom center line with the lane box
    if y2_car >= y1_lane and y2_car <= y2_lane:  # Car's bottom is within lane's vertical range
        # Calculate horizontal overlap
        overlap_left = max(bottom_center_left, x1_lane)
        overlap_right = min(bottom_center_right, x2_lane)

        if overlap_right > overlap_left:  # There is an overlap
            overlap_width = overlap_right - overlap_left
            bottom_center_area_width = bottom_center_right - bottom_center_left

            # Calculate percentage of bottom center area that overlaps with the lane
            overlap_percent = (overlap_width / bottom_center_area_width) * 100

            # If more than 30% of the bottom center area overlaps with the lane, it's a violation
            if overlap_percent >= 30:  # 30% threshold
                return True, overlap_percent

    # No violation detected
    return False, 0.0

# Process video for API endpoint
def process_video_api(video_path, violation_threshold_frames=12, cleanup=True):
    """Process video for illegal lane change detection

    Args:
        video_path: Path to the input video
        violation_threshold_frames: Number of consecutive frames to consider a sustained violation
        cleanup: Whether to remove the input video after processing

    Returns:
        Tuple containing (result_message, image_path) where:
        - result_message: "Illegal lane change violation detected" or "No violation"
        - image_path: Path to the saved image if violation detected, None otherwise
    """
    # Load the models
    lane_model = load_lane_model()
    car_model = load_car_model()

    print("Both lane detection and car detection models loaded successfully.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return "Error: Could not open video", None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize vehicle tracker with frame-based threshold
    vehicle_tracker = VehicleTracker(violation_threshold_frames=violation_threshold_frames)

    frame_count = 0
    violation_detected = False
    violation_image_path = None

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with vehicle tracking
        processed_frame, has_violations, violations_info, stop_processing = process_frame(
            frame, lane_model, car_model, frame_count, vehicle_tracker
        )

        # If a sustained violation is detected, save the frame and stop processing
        if stop_processing:
            # Generate a unique filename for the violation image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"violation_{timestamp}_frame{frame_count}.jpg"
            violation_image_path = os.path.join(outputs_dir, image_filename)

            # Add additional information to the frame before saving
            info_frame = processed_frame.copy()
            for violation in violations_info:
                if violation.get('sustained', False):
                    vehicle_id = violation.get('vehicle_id', 'unknown')
                    frames = vehicle_tracker.vehicles[vehicle_id]['consecutive_violation_frames']
                    info_text = f"Illegal Lane Change - Vehicle ID: {vehicle_id}"
                    cv2.putText(info_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save the frame
            cv2.imwrite(violation_image_path, info_frame)
            print(f"Sustained violation detected! Processing stopped at frame {frame_count}")
            print(f"Violation frame saved to {violation_image_path}")
            violation_detected = True
            break  # Stop processing the video

        # Print progress
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

    # Release resources
    cap.release()

    # Clean up the input file if requested
    if cleanup and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"Removed temporary file: {video_path}")
        except Exception as e:
            print(f"Failed to remove temporary file {video_path}: {e}")

    if violation_detected:
        return "Illegal lane change violation detected", violation_image_path
    else:
        return "No violation", None

# Original process_video function for command-line usage
def process_video(input_path, output_path, violation_threshold_frames=12):
    # Create violations folder if it doesn't exist
    violations_folder = "violations"
    if not os.path.exists(violations_folder):
        os.makedirs(violations_folder)
        print(f"Created folder: {violations_folder}")

    # Create finalviolation folder for saving the final violation frame
    finalviolation_folder = "finalviolation"
    if not os.path.exists(finalviolation_folder):
        os.makedirs(finalviolation_folder)
        print(f"Created folder: {finalviolation_folder}")

    # Load the models
    lane_model = load_lane_model()
    car_model = load_car_model()

    print("Both lane detection and car detection models loaded successfully.")

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize vehicle tracker with frame-based threshold
    vehicle_tracker = VehicleTracker(violation_threshold_frames=violation_threshold_frames)

    frame_count = 0
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Violation threshold: {violation_threshold_frames} frames")

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with vehicle tracking
        processed_frame, _, violations_info, _ = process_frame(
            frame, lane_model, car_model, frame_count, vehicle_tracker
        )

        # Write the processed frame to the output video
        out.write(processed_frame)

        # If any violation is detected, save the frame to the finalviolation folder and stop processing
        if len(violations_info) > 0:
            # Create a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            final_violation_filename = f"finalviolation/final_violation_{timestamp}.jpg"

            # Add additional information to the frame before saving
            info_frame = processed_frame.copy()
            for violation in violations_info:
                vehicle_id = violation.get('vehicle_id', 'unknown')
                frames = vehicle_tracker.vehicles[vehicle_id]['consecutive_violation_frames']
                info_text = f"Violation Detected - Vehicle ID: {vehicle_id}, Frames: {frames}"
                cv2.putText(info_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save the frame
            cv2.imwrite(final_violation_filename, info_frame)
            print(f"Violation detected! Processing stopped at frame {frame_count}")
            print(f"Violation frame saved to {final_violation_filename}")
            break  # Stop processing the video

        # Print progress
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete. Output saved to {output_path}")

# FastAPI endpoints
@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Process a video file to detect illegal lane changes.

    Args:
        file: The video file to process

    Returns:
        JSON response with the result and image path if violation detected
    """
    # Generate a unique filename for the uploaded video
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"

    # Save the uploaded file
    temp_file_path = os.path.join(current_dir, filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video synchronously - don't clean up the file so it can be viewed in the frontend
    result_message, image_path = process_video_api(
        temp_file_path,
        violation_threshold_frames=12,
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
    return {"message": "Illegal Lane Change Detection API is running"}

# Main function for testing
def main():
    input_path = "videos/IllegalLaneChange.mp4"
    output_path = "lane_and_car_detection_output.mp4"

    # Process video with 1 frame threshold for sustained violations
    process_video(input_path, output_path, violation_threshold_frames=12)

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8004)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
        main()
