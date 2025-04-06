import torch
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import uvicorn

# Create FastAPI app
app = FastAPI(title="License Plate Classification API", description="API for processing videos to detect and classify license plates")

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv5 models
print("Loading models...")

# License plate classification model
classification_model_path = "modelss/LicensePlateClassifier.pt"
classification_model = torch.hub.load('ultralytics/yolov5', 'custom', path=classification_model_path, force_reload=True)

# Car detection model
car_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
car_model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
car_model.conf = 0.3  # Lower threshold to detect more cars, we'll filter later
car_model.iou = 0.45

# License plate detection model
plate_model_path = "modelss/noPlate.pt"
plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path=plate_model_path, force_reload=True)
plate_model.conf = 0.3  # We'll check confidence in the code

# Define the classes for legal license plates
legal_plate_classes = {"punjab", "sindh", "KPK", "Islamabad", "balochistan"}

# Function to check if detections are legal license plates
def check_license_plate(image):
    # Run the model on the image
    results = classification_model(image)

    # Get the results dataframe
    df = results.pandas().xyxy[0]

    if len(df) == 0:
        return "Unknown", False, 0.0

    # Get the highest confidence detection
    highest_conf_idx = df['confidence'].idxmax()
    highest_conf = df.loc[highest_conf_idx, 'confidence']
    highest_conf_class = df.loc[highest_conf_idx, 'name']

    # Check if the confidence is high enough (60%)
    # if highest_conf < 0.6:
    #     return "Unknown", False, highest_conf

    # Check if it's a legal plate class
    if highest_conf_class in legal_plate_classes:
        return highest_conf_class, True, highest_conf

    return "Illegal", False, highest_conf

# Function to check if a car is completely in the frame
def is_car_in_frame(x1, y1, x2, y2, frame_width, frame_height, margin=10):
    # Check if the car is at least 'margin' pixels away from the frame edges
    return (x1 >= margin and y1 >= margin and
            x2 <= frame_width - margin and y2 <= frame_height - margin)

# Function to process a video
def process_video(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video", None

    frame_count = 0
    violation_image_path = None

    # Track cars with illegal plates
    car_tracking = {}
    next_car_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_with_detections = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # Detect cars in the frame
        car_results = car_model(frame)
        car_detections = car_results.pandas().xyxy[0]

        # Filter cars that are completely in the frame
        valid_cars = []
        for _, car in car_detections.iterrows():
            car_conf = float(car['confidence'])
            x1, y1, x2, y2 = int(car['xmin']), int(car['ymin']), int(car['xmax']), int(car['ymax'])

            # Check if car is completely in frame and has sufficient confidence
            if car_conf >= 0.4 and is_car_in_frame(x1, y1, x2, y2, frame_width, frame_height):
                valid_cars.append((car_conf, car))

        # Sort by confidence (highest first) and take top 5
        valid_cars.sort(reverse=True, key=lambda x: x[0])
        valid_cars = valid_cars[:5]  # Limit to 5 highest confidence cars

        # Current detected cars
        current_cars = {}

        # Process each car detection
        for car_conf, car in valid_cars:
            car_class = car['name']
            x1, y1, x2, y2 = int(car['xmin']), int(car['ymin']), int(car['xmax']), int(car['ymax'])
            car_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw bounding box around the car
            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame_with_detections, f"{car_class} {car_conf:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            # Try to match with existing tracked cars
            car_id = None
            for cid, car_info in car_tracking.items():
                prev_center = car_info['center']
                # Simple distance-based tracking
                distance = np.sqrt((car_center[0] - prev_center[0])**2 + (car_center[1] - prev_center[1])**2)
                if distance < 100:  # Threshold for considering it's the same car
                    car_id = cid
                    break

            # If no match found, create a new car ID
            if car_id is None:
                car_id = next_car_id
                next_car_id += 1
                car_tracking[car_id] = {
                    'center': car_center,
                    'illegal_count': 0,
                    'legal_count': 0,
                    'unknown_count': 0,
                    'last_seen': frame_count
                }
            else:
                # Update position
                car_tracking[car_id]['center'] = car_center
                car_tracking[car_id]['last_seen'] = frame_count

            # Extract car region
            car_region = frame[y1:y2, x1:x2].copy()  # Make a copy to avoid modifying the original frame
            if car_region.size == 0:
                continue

            # Detect license plates within the car region
            plate_results = plate_model(car_region)
            plate_detections = plate_results.pandas().xyxy[0]

            plate_class = None
            plate_legal = False
            plate_detected = False
            plate_conf = 0.0

            # Process each license plate detection
            for _, plate in plate_detections.iterrows():
                px1, py1, px2, py2 = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
                current_plate_conf = float(plate['confidence'])

                # Calculate absolute coordinates for the license plate in the original frame
                abs_px1, abs_py1 = x1 + px1, y1 + py1
                abs_px2, abs_py2 = x1 + px2, y1 + py2

                # Draw the license plate bounding box on the main frame
                cv2.rectangle(frame_with_detections, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 255, 0), 2)

                # Only process plates with confidence > 70%
                if current_plate_conf < 0.7:
                    cv2.putText(frame_with_detections, f"Plate {current_plate_conf:.2f} - Low conf",
                               (abs_px1, abs_py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    continue

                plate_detected = True

                # Extract license plate region
                plate_region = car_region[py1:py2, px1:px2]
                if plate_region.size == 0:
                    continue

                # Classify the license plate
                plate_class, plate_legal, plate_conf = check_license_plate(plate_region)

                # Determine color based on classification
                if plate_class == "Unknown":
                    color = (255, 165, 0)  # Orange for unknown
                    car_tracking[car_id]['unknown_count'] += 1
                elif plate_legal:
                    color = (0, 255, 0)  # Green for legal
                    car_tracking[car_id]['legal_count'] += 1
                else:
                    color = (0, 0, 255)  # Red for illegal
                    car_tracking[car_id]['illegal_count'] += 1

                # Draw the license plate classification on the main frame with thicker lines
                cv2.rectangle(frame_with_detections, (abs_px1, abs_py1), (abs_px2, abs_py2), color, 2)
                cv2.putText(frame_with_detections, f"{plate_class} {plate_conf:.2f}",
                           (abs_px1, abs_py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Only process the first valid plate
                break

            # If no plate detected with sufficient confidence, mark it
            if not plate_detected:
                cv2.putText(frame_with_detections, f"No valid plate detected",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            # Check if this car has consistently shown illegal plates
            is_illegal_car = False
            if (car_tracking[car_id]['illegal_count'] >= 5 and
                car_tracking[car_id]['illegal_count'] > car_tracking[car_id]['legal_count'] and
                car_tracking[car_id]['illegal_count'] > car_tracking[car_id]['unknown_count']):
                is_illegal_car = True

            # Update the car bounding box and text
            if is_illegal_car:
                color = (0, 0, 255)  # Red for confirmed illegal
                # Add text with violation count
                text = f"ILLEGAL PLATE - Confirmed ({car_tracking[car_id]['illegal_count']} frames)"
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_with_detections, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Add a title to the frame indicating violation
                title = "ILLEGAL LICENSE PLATE VIOLATION DETECTED"
                cv2.putText(frame_with_detections, title, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Add timestamp to the frame
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_with_detections, timestamp_str, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Save this frame as the violation image and stop processing
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"illegal_plate_{timestamp}_frame{frame_count}.jpg"
                violation_image_path = os.path.join(output_dir, image_filename)

                # Make sure the image is saved with high quality
                cv2.imwrite(violation_image_path, frame_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Stop processing as soon as we find a violation
                cap.release()
                return "Illegal license plate detected", violation_image_path
            else:
                # Determine color based on the most frequent classification
                if plate_class == "Unknown" or not plate_detected:
                    color = (255, 165, 0)  # Orange for unknown
                    text = f"Unknown plate ({car_tracking[car_id]['unknown_count']})"
                elif plate_legal:
                    color = (0, 255, 0)  # Green for legal
                    text = f"{plate_class} ({car_tracking[car_id]['legal_count']})"
                elif plate_class == "Illegal":
                    color = (0, 0, 255)  # Red for potential illegal
                    text = f"Potential illegal ({car_tracking[car_id]['illegal_count']}/5)"
                else:
                    color = (255, 255, 255)  # White for default
                    text = f"{car_class} {car_conf:.2f}"

                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_with_detections, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Mark the car as current
            current_cars[car_id] = True

        # Clean up tracking for cars not seen in this frame
        for car_id in list(car_tracking.keys()):
            if car_id not in current_cars and frame_count - car_tracking[car_id]['last_seen'] > 10:
                del car_tracking[car_id]

    # Release resources
    cap.release()

    # Return the result - no violation detected
    return "No illegal license plate detected", None

@app.post("/process-video/")
async def api_process_video(file: UploadFile = File(...)):
    """Process a video file to detect cars, license plates, and classify them.

    Args:
        file: The video file to process

    Returns:
        JSON response with violation details if detected
    """
    # Generate a unique filename for the uploaded video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"

    # Save the uploaded file
    temp_file_path = os.path.join(output_dir, filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process the video
        result_message, violation_image_path = process_video(temp_file_path)

        # Clean up the input file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Prepare the response
        response_data = {
            "message": result_message,
            "violation_detected": violation_image_path is not None
        }

        # Add image URL if violation detected
        if violation_image_path:
            image_filename = os.path.basename(violation_image_path)
            response_data["image_url"] = f"/images/{image_filename}"

        return JSONResponse(content=response_data)

    except Exception as e:
        # Clean up the input file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")



@app.get("/images/{filename}")
async def get_image(filename: str):
    """Get a violation image file.

    Args:
        filename: The name of the image file

    Returns:
        The image file
    """
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/jpeg"
    )

@app.get("/")
async def root():
    return {"message": "License Plate Classification API is running"}

# Run the FastAPI app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)