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
    def __init__(self, violation_threshold_frames=1):
        self.vehicles = defaultdict(lambda: {
            'is_violating': False,  # Boolean flag for current violation state
            'consecutive_violation_frames': 0,  # Count of consecutive frames in violation
            'last_position': None,
            'last_seen': 0,
            'sustained_violation': False
        })
        self.violation_threshold_frames = violation_threshold_frames
        self.cleanup_threshold = 20  # Remove vehicle if not seen for 20 frames

    def update_vehicle(self, car_box, frame_number, is_violating):
        x1, y1, x2, y2 = car_box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Find the closest vehicle from previous frames
        closest_id = None
        min_distance = 100  # Maximum pixel distance to consider same vehicle

        for vehicle_id, data in self.vehicles.items():
            if data['last_position'] is None:
                continue

            prev_center = data['last_position']
            distance = ((center[0] - prev_center[0]) ** 2 +
                       (center[1] - prev_center[1]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_id = vehicle_id

        # If no close match found, create new vehicle
        if closest_id is None:
            closest_id = len(self.vehicles)

        # Update vehicle data
        vehicle = self.vehicles[closest_id]
        vehicle['last_position'] = center
        vehicle['last_seen'] = frame_number

        # Update violation status
        if is_violating:
            # Set violation flag to true
            vehicle['is_violating'] = True
            # Increment consecutive violation frames counter
            vehicle['consecutive_violation_frames'] += 1
        else:
            # Reset violation status if car is no longer violating
            vehicle['is_violating'] = False
            vehicle['consecutive_violation_frames'] = 0

        # Check if violation threshold is reached
        if (vehicle['consecutive_violation_frames'] >= self.violation_threshold_frames and
            not vehicle['sustained_violation']):
            vehicle['sustained_violation'] = True
            return True, closest_id, vehicle

        return False, closest_id, vehicle

    def cleanup(self, current_frame):
        # Remove vehicles not seen recently
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

def is_car_on_lane(car_box, lane_points, margin=30, lane_contours=None, frame=None):
    """Check if a car is driving on a lane line using precise lane boundaries.

    Args:
        car_box: Tuple of (x1, y1, x2, y2) coordinates of the car
        lane_points: List of (x, y) coordinates forming the lane (for compatibility)
        margin: Margin of error in pixels
        lane_contours: List of contours representing precise lane boundaries
        frame: Optional frame for visualization

    Returns:
        bool: True if car is on lane, False otherwise
        float: Overlap percentage
        debug_frame: Frame with visualization (if frame is provided)
    """
    x1_car, y1_car, x2_car, y2_car = car_box
    car_width = x2_car - x1_car
    car_height = y2_car - y1_car

    # Create a debug frame if original frame is provided
    debug_frame = None
    if frame is not None:
        debug_frame = frame.copy()

    # Define the bottom center 60% area of the car
    center_x = (x1_car + x2_car) / 2
    bottom_center_width = car_width * 0.6  # 60% of car width
    bottom_center_left = int(center_x - bottom_center_width / 2)
    bottom_center_right = int(center_x + bottom_center_width / 2)

    # Create a mask for the bottom center area of the car
    # Use frame dimensions if available, otherwise make a reasonable estimate
    mask_height = frame.shape[0] if frame is not None else y2_car + 50
    mask_width = frame.shape[1] if frame is not None else x2_car + 50
    car_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    car_bottom_points = np.array([
        [bottom_center_left, y2_car-2],
        [bottom_center_right, y2_car-2],
        [bottom_center_right, y2_car+2],
        [bottom_center_left, y2_car+2]
    ], dtype=np.int32)
    cv2.fillPoly(car_mask, [car_bottom_points], 255)

    # Draw the car bottom area on debug frame
    if debug_frame is not None:
        cv2.fillPoly(debug_frame, [car_bottom_points], (0, 0, 255), 1)  # Red for car bottom area

    # If we have precise lane contours, use them for more accurate detection
    if lane_contours and len(lane_contours) > 0:
        max_overlap = 0.0
        is_violating = False
        violating_contour = None

        for contour in lane_contours:
            # Create a mask for the lane contour
            lane_mask = np.zeros_like(car_mask)
            cv2.drawContours(lane_mask, [contour], -1, 255, -1)  # Fill the contour

            # Find the intersection between car bottom center and lane
            intersection = cv2.bitwise_and(car_mask, lane_mask)
            intersection_area = cv2.countNonZero(intersection)

            if intersection_area > 0:
                # Calculate the car bottom center area
                car_area = cv2.countNonZero(car_mask)

                # Calculate overlap percentage
                if car_area > 0:
                    overlap_percent = (intersection_area / car_area) * 100

                    # If more overlap than previous contours
                    if overlap_percent > max_overlap:
                        max_overlap = overlap_percent
                        violating_contour = contour
                        if overlap_percent >= 50:  # 50% threshold for violation - more strict
                            is_violating = True

        # Draw the violating contour on debug frame
        if debug_frame is not None and violating_contour is not None:
            cv2.drawContours(debug_frame, [violating_contour], -1, (0, 255, 255), 2)  # Yellow for lane
            if is_violating:
                # Draw the car box in red for violation
                cv2.rectangle(debug_frame, (x1_car, y1_car), (x2_car, y2_car), (0, 0, 255), 2)
                # Add text showing overlap percentage
                cv2.putText(debug_frame, f"Violation: {max_overlap:.1f}%", (x1_car, y1_car - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return is_violating, max_overlap, debug_frame if frame is not None else None

    # Fallback to the original method if no contours are available
    if not lane_points:
        return False, 0.0, debug_frame if frame is not None else None

    # Check if any lane point is within the bottom center area
    for lane_x, lane_y in lane_points:
        # Check if the lane point is close to the car's bottom line
        if abs(lane_y - y2_car) <= margin:
            # Check if the lane point is within the bottom center area
            if bottom_center_left <= lane_x <= bottom_center_right:
                # Calculate overlap percentage based on how centered the point is
                distance_from_center = abs(lane_x - center_x)
                max_distance = bottom_center_width / 2
                overlap_percent = (1 - distance_from_center / max_distance) * 100

                # Draw the lane point on debug frame
                if debug_frame is not None:
                    cv2.circle(debug_frame, (int(lane_x), int(lane_y)), 5, (0, 255, 255), -1)

                return True, overlap_percent, debug_frame if frame is not None else None

    return False, 0.0, debug_frame if frame is not None else None

# Process a single frame to detect lane lines and cars
def process_frame(frame, lane_model, car_model, frame_number=None, vehicle_tracker=None):
    # Get frame dimensions for the lower center square calculation
    frame_height, frame_width = frame.shape[:2]

    # Define a trapezium-shaped detection zone (much broader at bottom and top)
    # Bottom coordinates (even broader at bottom toward both left and right)
    bottom_left_x = int(frame_width * 0.01)  # 5% from left (10% broader than before)
    bottom_right_x = int(frame_width * 0.99)  # 95% from left (10% broader than before)
    bottom_y = int(frame_height * 0.99)  # 99% from top (almost at the very bottom of the frame)

    # Top coordinates (broader than before, moved to the right)
    top_left_x = int(frame_width * 0.41)  # 36% from left (10% broader than before)
    top_right_x = int(frame_width * 0.47)  # 52% from left (10% broader than before)
    top_y = int(frame_height * 0.50)  # 50% from top (keep same height)

    # Define the trapezium as a polygon
    detection_zone = np.array([
        [bottom_left_x, bottom_y],   # Bottom left
        [bottom_right_x, bottom_y],  # Bottom right
        [top_right_x, top_y],        # Top right
        [top_left_x, top_y]          # Top left
    ], np.int32)

    # Convert frame to RGB for model input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = lane_model(frame_rgb)

    output_frame = frame.copy()
    detections = results.xyxy[0].cpu().numpy()

    # Draw only the border of the trapezium-shaped detection zone with more visibility
    cv2.polylines(output_frame, [detection_zone], True, (0, 255, 255), 2)
    cv2.putText(output_frame, "Detection Zone", (top_left_x, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    lane_boxes = []  # Store lane bounding boxes
    white_line_contours = []  # Kept for compatibility with existing code

    # First detect the solid lane lines using the model
    solid_lane_regions = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls == 1 and conf > 0.001:  # Only solid lines with confidence > 0.001
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the lane is in the trapezium-shaped detection zone
            # Create a rectangle for the lane
            lane_rect = np.array([
                [x1, y1],  # Top left
                [x2, y1],  # Top right
                [x2, y2],  # Bottom right
                [x1, y2]   # Bottom left
            ], np.int32)

            # Check if any corner of the lane rectangle is inside the detection zone
            lane_in_zone = False
            for point in lane_rect:
                if cv2.pointPolygonTest(detection_zone, (int(point[0]), int(point[1])), False) >= 0:
                    lane_in_zone = True
                    break

            if lane_in_zone:
                # Store the lane region for precise boundary detection
                solid_lane_regions.append((x1, y1, x2, y2))
                # Store the lane bounding box for compatibility
                lane_boxes.append((x1, y1, x2, y2))

    # Process each solid lane region to find precise boundaries
    for region in solid_lane_regions:
        x1, y1, x2, y2 = region

        # Extract the region of interest (ROI)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:  # Skip empty regions
            continue

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better white line detection
        # First enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_roi = clahe.apply(gray_roi)

        # Apply Gaussian blur to reduce noise
        blurred_roi = cv2.GaussianBlur(enhanced_roi, (5, 5), 0)

        # Use adaptive thresholding to better handle lighting variations
        thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # Apply morphological operations to clean up the result
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of the white lines
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:  # Increased minimum area threshold
                continue

            # Get the minimum area rectangle that fits the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Adjust contour coordinates to the original frame
            adjusted_contour = contour.copy()
            adjusted_contour[:,:,0] += x1
            adjusted_contour[:,:,1] += y1

            # Adjust box coordinates to the original frame
            adjusted_box = box.copy()
            adjusted_box[:,0] += x1
            adjusted_box[:,1] += y1

            # Draw the precise boundary of the solid lane
            color = (0, 255, 255)  # Yellow for solid line boundary
            cv2.drawContours(output_frame, [adjusted_contour], -1, color, 2)

            # Draw the minimum area rectangle around the solid lane
            cv2.drawContours(output_frame, [adjusted_box], 0, (0, 255, 0), 2)

            # Add a label for the solid lane
            cv2.putText(output_frame, "Solid Lane", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store the solid lane contour for violation detection
            white_line_contours.append(adjusted_contour)

    # Connect lane boxes to form continuous lanes
    left_lane, right_lane = connect_lane_boxes(lane_boxes, frame.shape)

    # Draw lanes on frame
    if left_lane:
        points = np.array(left_lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_frame, [points], False, (255, 0, 0), 2)

    if right_lane:
        points = np.array(right_lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_frame, [points], False, (0, 0, 255), 2)

    # Detect cars and check for violations
    output_frame, car_boxes = detect_cars(output_frame, car_model)
    violations_info = []

    for car_box in car_boxes:
        car_is_violating = False
        max_overlap_percent = 0
        x1_car, _, x2_car, y2_car = car_box
        car_width = x2_car - x1_car

        # Check if car is violating using precise lane contours with visualization
        is_violating, overlap_percent, debug_frame = is_car_on_lane(
            car_box, [], lane_contours=white_line_contours, frame=frame
        )

        if is_violating:
            car_is_violating = True
            max_overlap_percent = overlap_percent

            # Save the debug frame with violation visualization
            if debug_frame is not None and frame_number is not None:
                # Create violations folder if it doesn't exist
                violations_folder = "violations"
                if not os.path.exists(violations_folder):
                    os.makedirs(violations_folder)

                # Save the frame with timestamp and frame number
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                violation_filename = f"{violations_folder}/violation_{timestamp}_frame{frame_number}.jpg"
                cv2.imwrite(violation_filename, debug_frame)
                print(f"Saved violation frame to {violation_filename}")

        # Fallback to traditional method if no violation detected
        if not car_is_violating:
            for lane_points in [left_lane, right_lane]:
                is_violating, overlap_percent, _ = is_car_on_lane(car_box, lane_points)
                if is_violating:
                    car_is_violating = True
                    max_overlap_percent = max(max_overlap_percent, overlap_percent)

        # Update vehicle tracker
        if vehicle_tracker is not None:
            _, vehicle_id, vehicle_data = vehicle_tracker.update_vehicle(
                car_box, frame_number, car_is_violating
            )

            # Draw violation status
            if vehicle_data['sustained_violation']:
                color = (0, 0, 255)  # Red
                text = f"Sustained Violation ({vehicle_data['consecutive_violation_frames']} frames)"
            elif car_is_violating:
                color = (0, 0, 255)  # Red for all violations
                text = f"Violation: {vehicle_data['consecutive_violation_frames']} frames"
            else:
                color = (0, 255, 0)  # Green
                text = f"Tracking ID: {vehicle_id}"

            x1, y1, x2, y2 = car_box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if car_is_violating or vehicle_data['sustained_violation']:
                violations_info.append({
                    'frame_number': frame_number,
                    'vehicle_id': vehicle_id,
                    'car_box': car_box,
                    'overlap_percent': max_overlap_percent,
                    'consecutive_violation_frames': vehicle_data['consecutive_violation_frames'],
                    'sustained': vehicle_data['sustained_violation']
                })

    # Cleanup tracker
    if vehicle_tracker is not None:
        vehicle_tracker.cleanup(frame_number)

    # Check if any vehicle has a sustained violation
    stop_processing = False
    for violation in violations_info:
        if violation.get('sustained', False):
            stop_processing = True
            break

    return output_frame, len(violations_info) > 0, violations_info, stop_processing

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
def process_video_api(video_path, violation_threshold_frames=1, cleanup=True):
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
                    info_text = f"Illegal Lane Change - Vehicle ID: {vehicle_id}, Frames: {frames}"
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
def process_video(input_path, output_path, violation_threshold_frames=1):
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
        violation_threshold_frames=1,
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
    process_video(input_path, output_path, violation_threshold_frames=1)

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8004)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
        main()