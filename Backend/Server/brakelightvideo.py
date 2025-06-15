import cv2
import torch
import numpy as np
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uuid
from datetime import datetime

app = FastAPI(title="Brake Light Detection API", description="API for processing videos to detect broken brake lights")

# Create outputs directory
outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Set correct paths to model files
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, 'modelss', 'yolov5s.pt')

# Verify model path exists
if not os.path.exists(yolov5_path):
    raise FileNotFoundError(f"YOLOv5 model not found at: {yolov5_path}")

# Load YOLOv5 model (detect only cars)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_path)
model.conf = 0.4
model.classes = [2]  # Only detect cars

def detect_brake_lights(frame, car_bbox):
    """
    Detect brake lights in a car's region of interest

    Args:
        frame: Input frame
        car_bbox: Car bounding box (x1, y1, x2, y2)

    Returns:
        tuple: (brake_light_count, is_violation, annotated_roi)
    """
    x1, y1, x2, y2 = car_bbox

    # Create ROI focusing on the rear of the car
    height_car = y2 - y1
    shrink_y1 = y1 + int(0.3 * height_car)
    shrink_y2 = y2 - int(0.3 * height_car)

    roi = frame[shrink_y1:shrink_y2, x1:x2]
    if roi.size == 0:
        return 0, False, None

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define red color ranges for brake lights
    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 60, 40])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    brake_light_count = 0
    best_threshold = None
    annotated_roi = roi.copy()

    # Try different brightness thresholds to detect brake lights
    for threshold in range(130, 201):
        v = hsv[:, :, 2]
        bright_mask = cv2.inRange(v, threshold, 255)
        final_mask = cv2.bitwise_and(red_mask, bright_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_roi, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            count += 1

        if count > 0:
            brake_light_count = count
            best_threshold = threshold
            break  # Found brake lights at this threshold, stop checking higher ones

    # Determine if this is a violation (broken brake lights)
    # Conservative approach: mark as broken only if no brake lights detected or only 1 detected
    is_violation = brake_light_count < 2

    return brake_light_count, is_violation, annotated_roi

def process_image_brake_lights(image_path, cleanup=True):
    """
    Process image for brake light detection

    Args:
        image_path: Path to the input image
        cleanup: Whether to remove the input image after processing

    Returns:
        Tuple containing (result_message, output_image_path, detections_count) where:
        - result_message: "Broken brake lights detected" or "No violation"
        - output_image_path: Path to the saved processed image
        - detections_count: Number of cars with brake light detections
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return "Error: Could not read image", None, 0

    h, w = frame.shape[:2]

    # Define detection zone (lower half of the frame)
    zone_top = h // 2
    zone_bottom = h
    zone_left = int(w * 0.1)
    zone_right = int(w * 0.9)

    # Detect cars in the image
    results = model(frame)
    cars = results.xyxy[0].cpu().numpy()

    violation_detected = False
    detections_count = 0

    for car in cars:
        x1, y1, x2, y2, conf, cls = map(int, car[:6])
        car_area = (x2 - x1) * (y2 - y1)

        # Check if car is in the detection zone
        inter_left = max(x1, zone_left)
        inter_right = min(x2, zone_right)
        inter_top = max(y1, zone_top)
        inter_bottom = min(y2, zone_bottom)
        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

        if inter_area / car_area < 0.6:
            continue

        detections_count += 1

        # Detect brake lights for this car
        brake_light_count, is_violation, annotated_roi = detect_brake_lights(frame, (x1, y1, x2, y2))

        # Draw car bounding box and brake light info
        color = (0, 0, 255) if is_violation else (0, 255, 0)  # Red for violation, green for good
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add text labels
        status = "BROKEN" if is_violation else "GOOD"
        cv2.putText(frame, f'Brake Lights: {status} ({brake_light_count})',
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If any car has a violation, mark as violation detected
        if is_violation:
            violation_detected = True

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"output_{timestamp}_{unique_id}_{os.path.basename(image_path)}"
    output_image_path = os.path.join(outputs_dir, output_filename)

    # Save the processed image
    cv2.imwrite(output_image_path, frame)

    # Clean up the input file if requested
    if cleanup and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Removed temporary file: {image_path}")
        except Exception as e:
            print(f"Failed to remove temporary file {image_path}: {e}")

    if violation_detected:
        return "Broken brake lights detected", output_image_path, detections_count
    else:
        return "No violation detected", output_image_path, detections_count


@app.post("/detect-brake-lights/")
async def detect_brake_lights_endpoint(file: UploadFile = File(...)):
    """
    Process an image file to detect broken brake lights.

    Args:
        file: The image file to process

    Returns:
        JSON response with the result and processed image
    """
    # Generate a unique filename for the uploaded image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"

    # Save the uploaded file
    temp_file_path = os.path.join(current_dir, filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image synchronously
    result_message, output_image_path, detections_count = process_image_brake_lights(
        temp_file_path,
        cleanup=True
    )

    # Prepare the response
    response_data = {
        "message": result_message,
        "detections_count": detections_count,
    }

    if output_image_path:
        # Get just the filename from the path
        image_filename = os.path.basename(output_image_path)
        response_data["image_url"] = f"/images/{image_filename}"

        # Check if violation was detected
        if "Broken brake lights detected" in result_message:
            response_data["violation_detected"] = True
        else:
            response_data["violation_detected"] = False
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
    return {"message": "Brake Light Detection API is running"}


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8006)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
