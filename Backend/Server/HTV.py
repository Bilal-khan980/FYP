import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uuid
from datetime import datetime

app = FastAPI(title="HTV Lane Detection API", description="API for processing videos to detect vehicles in lanes")

# Create outputs directory
outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Set correct paths to model files based on actual file locations
current_dir = os.path.dirname(os.path.abspath(__file__))
best_weights_path = os.path.join(current_dir, 'modelss', 'laneDetecion.pt')
yolov5m_path = os.path.join(current_dir, 'yolov5(1)', 'yolov5m.pt')

# Verify paths exist
if not os.path.exists(best_weights_path):
    raise FileNotFoundError(f"Lane detection model not found at: {best_weights_path}")
if not os.path.exists(yolov5m_path):
    raise FileNotFoundError(f"YOLOv5m model not found at: {yolov5m_path}")

detectRightLane = True

# Load both models properly
vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5m_path)
lane_model = torch.hub.load('ultralytics/yolov5', 'custom', path=best_weights_path, force_reload=True)

def visualize_vehicles(image, vehicles, lane_vehicles, lane_points, display=False):
    marked_image = image.copy()
    best_line, corner_points = lane_points
    lane_polygon = np.array([
        [best_line[0][0], best_line[0][1]],
        [best_line[1][0], best_line[1][1]],
        [corner_points[1][0], corner_points[1][1]],
        [corner_points[0][0], corner_points[0][1]]
    ], np.int32)

    overlay = marked_image.copy()
    cv2.fillPoly(overlay, [lane_polygon], color=(0, 255, 0))
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, marked_image, 1 - alpha, 0, marked_image)
    cv2.polylines(marked_image, [lane_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    for vehicle in vehicles:
        x1, y1, x2, y2, conf, cls = vehicle
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(conf)
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.putText(marked_image, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    for vehicle in lane_vehicles:
        x1, y1, x2, y2, conf, cls = vehicle
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(conf)
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(marked_image, 'Lane Vehicle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return marked_image


def check_vehicles_in_lane(vehicles, lane_points):
    best_line, corner_points = lane_points
    lane_polygon = np.array([
        [best_line[0][0], best_line[0][1]],
        [best_line[1][0], best_line[1][1]],
        [corner_points[1][0], corner_points[1][1]],
        [corner_points[0][0], corner_points[0][1]]
    ])
    lane_path = Path(lane_polygon)
    vehicles_in_lane = []

    for vehicle in vehicles:
        x1, y1, x2, y2, conf, cls = vehicle
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        bottom_corners = [(x1, y2), (x2, y2)]
        if any(lane_path.contains_point(point) for point in bottom_corners):
            vehicles_in_lane.append(vehicle)

    return np.array(vehicles_in_lane)


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


def detect_vehicles(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    vehicles = []

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if cls in [2, 5, 7]:
            vehicles.append([float(x1), float(y1), float(x2), float(y2), float(conf), coco_classes[int(cls)]])

    return np.array(vehicles)


def process_detections(image, model, filterRightPoints=detectRightLane):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_center = image.shape[1] // 2
    results = model(image_rgb)
    detections = []

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        if cls == 1:
            if filterRightPoints and center_x < img_center:
                continue
            elif not filterRightPoints and center_x >= img_center:
                continue

        detections.append([center_x, center_y, width, height, conf, cls])

    return image_rgb, np.array(detections)


def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return angle


def point_to_line_distance(point, line_p1, line_p2):
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return numerator/denominator


def find_and_connect_nearest_lane(image, detections, notShowOutput=False, detectFarSolidPoint=not detectRightLane):
    frame = image.copy()
    dotted_points = []
    solid_points = []
    solid_boxes = []

    for det in detections:
        x, y, w, h, conf, cls = det
        if cls == 0:
            dotted_points.append((int(x), int(y)))
        else:
            solid_points.append((int(x), int(y)))
            solid_boxes.append((int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)))

    if not solid_points or len(dotted_points) < 2:
        if not notShowOutput:
            print("Not enough solid points or dotted points found.")
        return frame, None, None

    dotted_points = np.array(dotted_points)
    solid_points = np.array(solid_points)
    solid_boxes = np.array(solid_boxes)

    ref_idx = solid_points[:, 0].argmax() if detectFarSolidPoint else solid_points[:, 0].argmin()
    boundary_solid_point = solid_points[ref_idx]
    ref_box = solid_boxes[ref_idx]

    img_center = image.shape[1] // 2
    if boundary_solid_point[0] >= img_center:
        corner_points = [
            (ref_box[0], ref_box[1]),
            (ref_box[2], ref_box[3])
        ]
    else:
        corner_points = [
            (ref_box[2], ref_box[1]),
            (ref_box[0], ref_box[3])
        ]

    min_distance = float('inf')
    best_line = None
    y_sorted_dots = dotted_points[dotted_points[:, 1].argsort()]

    for i in range(len(y_sorted_dots)-1):
        for j in range(i+1, len(y_sorted_dots)):
            p1, p2 = y_sorted_dots[i], y_sorted_dots[j]

            if abs(p2[1] - p1[1]) < image.shape[0]/10:
                continue

            angle = calculate_angle(p1, p2)
            if angle < 30 or angle > 55:
                continue

            dist = point_to_line_distance(boundary_solid_point, p1, p2)
            if dist < min_distance:
                min_distance = dist
                best_line = (p1, p2)

    if best_line is None:
        if not notShowOutput:
            print("No suitable lines found.")
        return frame, None, None

    for point in dotted_points:
        cv2.circle(frame, tuple(point), 3, (0,0,255), -1)

    for point in solid_points:
        cv2.circle(frame, tuple(point), 3, (255,0,0), -1)

    cv2.circle(frame, tuple(boundary_solid_point), 5, (0,255,255), -1)

    for corner in corner_points:
        cv2.circle(frame, tuple(map(int, corner)), 4, (255,0,255), -1)
        cv2.line(frame, tuple(map(int, corner)), tuple(boundary_solid_point), (255,0,255), 2)

    p1, p2 = best_line
    cv2.circle(frame, tuple(p1), 4, (0,255,0), -1)
    cv2.circle(frame, tuple(p2), 4, (0,255,0), -1)
    cv2.line(frame, tuple(p1), tuple(p2), (0,255,255), 2)
    if notShowOutput:
        return frame, best_line, corner_points

    plt.figure(figsize=(12,8))
    plt.imshow(image)

    plt.scatter(dotted_points[:,0], dotted_points[:,1],
                c='red', s=20, alpha=0.5, label='Dotted Points')

    plt.scatter(solid_points[:,0], solid_points[:,1],
                c='blue', s=20, label='Solid Points')

    plt.scatter(boundary_solid_point[0], boundary_solid_point[1],
                c='yellow', s=100, label='Reference Point')
    corner_points = np.array(corner_points)
    plt.scatter(corner_points[:,0], corner_points[:,1],
                c='magenta', s=50, label='Corner Points')

    plt.plot([corner_points[0,0], boundary_solid_point[0]],
             [corner_points[0,1], boundary_solid_point[1]],
             'm-', linewidth=2)
    plt.plot([corner_points[1,0], boundary_solid_point[0]],
             [corner_points[1,1], boundary_solid_point[1]],
             'm-', linewidth=2)

    p1, p2 = best_line
    plt.scatter([p1[0], p2[0]], [p1[1], p2[1]],
                c='green', s=50, label='Selected Points')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
             'y-', linewidth=2, label=f'Lane Line (dist={min_distance:.1f})')

    plt.legend()
    plt.axis('off')
    plt.show()

    return frame, best_line, corner_points

def process_video_lanes_and_lane_vehicles(video_path, model, vehicle_model, cleanup=True):
    """Process video for lane detection and visualization

    Args:
        video_path: Path to the input video
        model: Lane detection model
        vehicle_model: Vehicle detection model
        cleanup: Whether to remove the input video after processing

    Returns:
        Tuple containing (result_message, image_path) where:
        - result_message: "HTV in first lane detected" or "No violation"
        - image_path: Path to the saved image if violation detected, None otherwise
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return "Error: Could not open video", None

    frame_count = 0
    violation_detected = False
    violation_image_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = []
        _, detections = process_detections(frame, model)
        marked_frame, best_line, corner_points = find_and_connect_nearest_lane(frame, detections, notShowOutput=True)

        if best_line is None or corner_points is None:
            continue

        vehicles = detect_vehicles(frame, vehicle_model)
        lane_points = [best_line, corner_points]
        lane_vehicles = check_vehicles_in_lane(vehicles, lane_points)

        # If lane vehicles detected, save the frame and stop processing
        if len(lane_vehicles) > 0:
            # Generate a unique filename for the violation image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"violation_{timestamp}_frame{frame_count}.jpg"
            violation_image_path = os.path.join(outputs_dir, image_filename)

            # Draw the vehicles and lane on the frame
            marked_frame = visualize_vehicles(frame, vehicles, lane_vehicles, lane_points, display=False)

            # Save the image
            cv2.imwrite(violation_image_path, marked_frame)
            violation_detected = True
            break

    cap.release()

    # Clean up the input file if requested
    if cleanup and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"Removed temporary file: {video_path}")
        except Exception as e:
            print(f"Failed to remove temporary file {video_path}: {e}")

    if violation_detected:
        return "HTV in first lane detected", violation_image_path
    else:
        return "No violation", None



@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Process a video file to detect lanes and vehicles.

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

    # Process the video synchronously (not in background)
    result_message, image_path = process_video_lanes_and_lane_vehicles(
        temp_file_path,
        lane_model,
        vehicle_model,
        cleanup=True
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
    return {"message": "HTV Lane Detection API is running"}



if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
