import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uuid
from datetime import datetime
import shutil
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="License Plate Detection and Classification API",
    description="API for detecting and classifying license plates in images",
    version="1.0.0"
)

# Create directories for uploads and outputs
current_dir = os.path.dirname(os.path.abspath(__file__))
uploads_folder = os.path.join(current_dir, "uploads")
output_dir = os.path.join(current_dir, "outputs")
os.makedirs(uploads_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Define model paths
model_dir = os.path.join(current_dir, "modelss")
car_model_path = 'yolov5s.pt'
plate_model_path = os.path.join(model_dir, "noPlate.pt")
classification_model_path = os.path.join(model_dir, "LicensePlateClassifier.pt")

print("Loading models...")

# Load car detection model (YOLOv5s)
if not os.path.exists(car_model_path):
    print(f"Warning: Car model not found at {car_model_path}. Using pretrained model.")
    # Use local model path instead of downloading from torch hub
    yolov5_dir = os.path.join(current_dir, '..', '..', 'yolov5(1)')
    if os.path.exists(yolov5_dir):
        print(f"Loading YOLOv5 from local directory: {yolov5_dir}")
        car_model = torch.hub.load(yolov5_dir, 'custom', path='yolov5s.pt', source='local')
    else:
        print(f"Local YOLOv5 directory not found at {yolov5_dir}, using torch hub")
        car_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
else:
    car_model = torch.hub.load('ultralytics/yolov5', 'custom', path=car_model_path, force_reload=False)

# Set car model parameters
car_model.classes = [2, 5, 7]  # car, bus, truck
car_model.conf = 0.5  # Car detection confidence

# Load license plate detection model
if not os.path.exists(plate_model_path):
    print(f"Error: License plate model not found at {plate_model_path}")
    plate_model = None
else:
    # Use local model path
    yolov5_dir = os.path.join(current_dir, '..', '..', 'yolov5(1)')
    if os.path.exists(yolov5_dir):
        print(f"Loading plate model from local directory: {yolov5_dir}")
        plate_model = torch.hub.load(yolov5_dir, 'custom', path=plate_model_path, source='local')
    else:
        plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path=plate_model_path, force_reload=False)
    plate_model.conf = 0.5  # License plate detection confidence

# Load license plate classification model
if not os.path.exists(classification_model_path):
    print(f"Error: Classification model not found at {classification_model_path}")
    classification_model = None
else:
    # Use local model path
    yolov5_dir = os.path.join(current_dir, '..', '..', 'yolov5(1)')
    if os.path.exists(yolov5_dir):
        print(f"Loading classification model from local directory: {yolov5_dir}")
        classification_model = torch.hub.load(yolov5_dir, 'custom', path=classification_model_path, source='local')
    else:
        classification_model = torch.hub.load('ultralytics/yolov5', 'custom', path=classification_model_path, force_reload=False)

# Define legal plate classes and confidence thresholds
legal_plate_classes = {"punjab", "sindh", "KPK", "Islamabad", "balochistan"}
CLASSIFICATION_THRESHOLD = 0.5  # Classification confidence threshold

def check_license_plate(image):
    """
    Classify a license plate image.
    Returns:
        tuple: (class_name, is_legal, confidence, status)
        status can be 'legal' or 'illegal'
    """
    # Check if classification model is available
    if classification_model is None:
        return None, False, 0.0, "illegal"

    try:
        results = classification_model(image)
        df = results.pandas().xyxy[0]

        if len(df) == 0:
            return None, False, 0.0, "illegal"

        # Get highest confidence detection
        highest_conf_idx = df['confidence'].idxmax()
        highest_conf = df.loc[highest_conf_idx, 'confidence']
        highest_conf_class = df.loc[highest_conf_idx, 'name']

        # If it's one of the classes in the model, mark as legal
        # Otherwise mark as illegal
        if highest_conf_class in legal_plate_classes:
            return highest_conf_class, True, highest_conf, "legal"
        else:
            return highest_conf_class, False, highest_conf, "illegal"

    except Exception as e:
        print(f"Error classifying license plate: {str(e)}")
        return None, False, 0.0, "illegal"

def process_image(input_path, output_path):
    """
    Process an image to detect cars, license plates, and classify them.

    Args:
        input_path: Path to input image
        output_path: Path to output image

    Returns:
        bool: True if processing was successful, False otherwise
        dict: Information about detections and classifications
    """
    # Read the image
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not open image file {input_path}")
        return False, {"error": "Could not open image file"}

    print(f"Processing image: {input_path}")
    print(f"Output will be saved to: {output_path}")

    try:
        # Create a copy of the frame for processing
        output_frame = frame.copy()

        # Detect cars in the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        car_results = car_model(frame_rgb)
        car_detections = car_results.pandas().xyxy[0]

        detections_info = []

        # Process each car detection
        for _, car in car_detections.iterrows():
            # Get car bounding box
            x1, y1, x2, y2 = map(int, [car['xmin'], car['ymin'], car['xmax'], car['ymax']])
            car_conf = float(car['confidence'])
            car_class = car['name']

            # Draw car bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, f"{car_class}: {car_conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract car region for license plate detection
            car_region = frame[y1:y2, x1:x2].copy()
            if car_region.size == 0 or car_region.shape[0] == 0 or car_region.shape[1] == 0:
                continue

            # Detect license plate in the car region
            plate_results = plate_model(car_region)
            plate_detections = plate_results.pandas().xyxy[0]

            # Process each license plate detection
            for _, plate in plate_detections.iterrows():
                # Get plate bounding box (relative to car region)
                px1, py1, px2, py2 = map(int, [plate['xmin'], plate['ymin'], plate['xmax'], plate['ymax']])
                plate_conf = float(plate['confidence'])

                # Calculate absolute coordinates in the original frame
                abs_px1, abs_py1 = x1 + px1, y1 + py1
                abs_px2, abs_py2 = x1 + px2, y1 + py2

                # Extract and classify plate
                plate_region = car_region[py1:py2, px1:px2]
                if plate_region.size > 0:
                    plate_class, is_legal, class_conf, status = check_license_plate(plate_region)

                    # Determine color based on status
                    if status == "legal":
                        color = (0, 255, 0)  # Green for legal
                    else:  # illegal
                        color = (0, 0, 255)  # Red for illegal

                    # Draw license plate bounding box
                    cv2.rectangle(output_frame, (abs_px1, abs_py1), (abs_px2, abs_py2), color, 2)

                    # Prepare label text
                    if plate_class and status == "legal":
                        label = f"{plate_class}: {class_conf:.2f}"
                    else:
                        label = f"illegal: {class_conf:.2f}"

                    # Draw label
                    cv2.putText(output_frame, label, (abs_px1, abs_py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Save detection information
                    detections_info.append({
                        'car_box': [x1, y1, x2, y2],
                        'car_class': car_class,
                        'car_confidence': car_conf,
                        'plate_box': [abs_px1, abs_py1, abs_px2, abs_py2],
                        'plate_class': plate_class,
                        'plate_confidence': class_conf,
                        'status': status,
                        'is_legal': is_legal
                    })

        # Save the processed image
        cv2.imwrite(output_path, output_frame)

        # Prepare result information
        result_info = {
            "detections_count": len(detections_info),
            "detections": detections_info,
            "output_path": output_path
        }

        return True, result_info

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False, {"error": f"An error occurred: {str(e)}"}

@app.post("/detect-license-plate/", summary="Detect and classify license plates in an image",
         description="Upload an image file to detect cars and license plates, and classify them as legal or illegal")
async def detect_license_plate(file: UploadFile = File(..., description="Image file to process")):
    try:
        # Check file extension
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return {"error": "Unsupported file format. Please upload an image file (jpg, jpeg, png, bmp)."}

        # Create a unique filename for the uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"input_{timestamp}_{unique_id}_{file.filename}"
        temp_input = os.path.join(uploads_folder, input_filename)

        # Save uploaded file
        with open(temp_input, "wb") as buffer:
            buffer.write(await file.read())

        # Process image
        output_filename = f"output_{timestamp}_{unique_id}_{file.filename}"
        output_path = os.path.join(output_dir, output_filename)
        success, result_info = process_image(temp_input, output_path)

        if not success:
            return {"error": result_info.get("error", "Failed to process image")}

        # Add image URL to result
        result_info["image_url"] = f"/images/{output_filename}"

        return JSONResponse(content=result_info)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Get a processed image file.

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
    return {"message": "License Plate Detection and Classification API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
