import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
import os

# Fix: Use correct path with 'modelss' instead of 'models'
current_dir = os.path.dirname(os.path.abspath(__file__))
best_weights_path = os.path.join(current_dir, '..', 'yolov5(1)', 'modelss', 'laneDetecion.pt')
yolov5m_path = os.path.join(current_dir, '..', 'yolov5(1)', 'yolov5m.pt')

# Verify paths exist
if not os.path.exists(best_weights_path):
    raise FileNotFoundError(f"Lane detection model not found at: {best_weights_path}")
if not os.path.exists(yolov5m_path):
    raise FileNotFoundError(f"YOLOv5m model not found at: {yolov5m_path}")

detectRightLane = True

# Load both models properly
vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5m_path)
lane_model = torch.hub.load('ultralytics/yolov5', 'custom', path=best_weights_path, force_reload=True)

# Update the video paths with absolute paths
video_path = "htv.mp4"
output_path = os.path.join(current_dir,"out_htv.mp4")

# Create videos directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Verify video exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found at: {video_path}")

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

def process_video_lanes_and_lane_vehicles(video_path, model,vehicle_model, output_path=None):
    """Process video for lane detection and visualization"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, detections = process_detections(frame, model)
        
        
        results = model(frame_rgb)
        marked_frame,best_line, corner_points=find_and_connect_nearest_lane(frame, detections,notShowOutput=True)
        vehicles = detect_vehicles(frame, vehicle_model)
        if best_line is None or corner_points is None:
            out.write(marked_frame)
            continue
        lane_points = [best_line, corner_points]
        lane_vehicles = check_vehicles_in_lane(vehicles, lane_points)

        marked_frame=visualize_vehicles(frame, vehicles, lane_vehicles, lane_points,display=False)
        if output_path:
            out.write(marked_frame)
        else:
            plt.imshow(marked_frame) 
            plt.title('Lane Detection')
            plt.axis('off') 
            plt.show()

    cap.release()
    if output_path:
        out.release()



print('Processing Video')
# Pass the lane_model instead of the path
process_video_lanes_and_lane_vehicles(video_path, lane_model, vehicle_model, output_path)
print('Video Processed Successfully.')
print(f'Saved At: {output_path} ')

