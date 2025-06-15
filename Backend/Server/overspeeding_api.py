#!/usr/bin/env python3
"""
FastAPI Overspeeding Detection API
Integrates with FYP frontend for traffic violation detection
"""

import cv2
import numpy as np
import os
import sys
import shutil
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

current_dir = os.path.dirname(os.path.abspath(__file__))
overspeeding_dir = os.path.join(current_dir, 'overspeeding')
sys.path.append(overspeeding_dir)

try:
    from dashcam_speed_detector import DashcamSpeedDetector
except ImportError as e:
    print(f"Error importing DashcamSpeedDetector: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Overspeeding directory: {overspeeding_dir}")
    print(f"Python path: {sys.path}")
    raise

app = FastAPI(title="Overspeeding Detection API", description="API for detecting overspeeding violations in dashcam videos")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create outputs directory
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Speed limit configuration
SPEED_LIMIT_KMH = 10  # Speed limit in km/h

# Initialize the detector globally
detector = None

def initialize_detector():
    """Initialize the dashcam speed detector"""
    global detector
    try:
        print("Initializing overspeeding detector...")
        detector = DashcamSpeedDetector(
            pwc_model_path="pwc_net.pth",  # Will be disabled automatically
            yolo_model_path="yolov8n.pt",
            speed_model_type="nvidia",
            device="cpu"
        )
        
        # Configure for fast violation detection
        detector.speed_window_size = 5  # Smaller window for faster response
        detector.min_flow_threshold = 0.3
        detector.speed_change_threshold = 10.0  # Allow larger changes for faster detection
        detector.speed_limit = SPEED_LIMIT_KMH  # Set the speed limit for vehicle tracking
        
        print("✅ Overspeeding detector initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

def process_video_overspeeding(video_path: str, cleanup: bool = True):
    """
    Process video for overspeeding detection of OTHER vehicles (not ego vehicle)

    Args:
        video_path: Path to the input video
        cleanup: Whether to remove the input video after processing

    Returns:
        Tuple containing (result_message, image_path)
    """
    global detector

    if detector is None:
        if not initialize_detector():
            return "Error: Could not initialize detector", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return "Error: Could not open video", None

    frame_count = 0
    violation_detected = False
    violation_image_path = None
    violating_vehicle_id = None
    violating_vehicle_speed = 0

    print(f"🎬 Processing video for OTHER vehicle overspeeding detection...")
    print(f"📏 Speed limit: {SPEED_LIMIT_KMH} km/h")
    print(f"🚗 Monitoring other vehicles in the scene...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame for speed detection
            try:
                processed_frame, ego_speed, vehicles = detector.process_frame(frame)

                # Check each tracked vehicle for overspeeding
                violation_found = False
                for vehicle in vehicles:
                    track_id = vehicle['track_id']
                    vehicle_speed = vehicle.get('speed', 0)

                    # Check if this vehicle is overspeeding
                    if vehicle_speed > SPEED_LIMIT_KMH:
                        print(f"🚨 OTHER VEHICLE OVERSPEEDING DETECTED!")
                        print(f"   Frame: {frame_count}")
                        print(f"   Vehicle ID: {track_id}")
                        print(f"   Vehicle Speed: {vehicle_speed:.1f} km/h")
                        print(f"   Speed Limit: {SPEED_LIMIT_KMH} km/h")
                        print(f"   Violation: {vehicle_speed - SPEED_LIMIT_KMH:.1f} km/h over limit")
                        print(f"   Ego Speed: {ego_speed:.1f} km/h (for reference)")

                        # Generate unique filename for violation image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = f"vehicle_overspeeding_{timestamp}_frame{frame_count}_vehicle{track_id}.jpg"
                        violation_image_path = os.path.join(outputs_dir, image_filename)

                        # Add violation information to the frame
                        violation_frame = processed_frame.copy()

                        # Add red overlay for violation
                        overlay = violation_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (violation_frame.shape[1], 120), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.3, violation_frame, 0.7, 0, violation_frame)

                        # Add violation text
                        cv2.putText(violation_frame, "VEHICLE OVERSPEEDING VIOLATION!",
                                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        cv2.putText(violation_frame, f"Vehicle ID: {track_id} Speed: {vehicle_speed:.1f} km/h",
                                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(violation_frame, f"Speed Limit: {SPEED_LIMIT_KMH} km/h | Violation: +{vehicle_speed - SPEED_LIMIT_KMH:.1f} km/h",
                                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Highlight the violating vehicle with a red box
                        if 'bbox' in vehicle:
                            x1, y1, x2, y2 = vehicle['bbox']
                            cv2.rectangle(violation_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                            cv2.putText(violation_frame, f"OVERSPEEDING: {vehicle_speed:.1f} km/h",
                                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Save the violation frame
                        cv2.imwrite(violation_image_path, violation_frame)
                        violation_detected = True
                        violating_vehicle_id = track_id
                        violating_vehicle_speed = vehicle_speed
                        violation_found = True

                        print(f"💾 Violation frame saved: {violation_image_path}")
                        break  # Stop processing immediately after first violation

                if violation_found:
                    break

                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    active_vehicles = len([v for v in vehicles if v.get('speed', 0) > 0])
                    print(f"📊 Processed {frame_count} frames, tracking {active_vehicles} vehicles, ego speed: {ego_speed:.1f} km/h")

            except Exception as e:
                print(f"⚠️ Error processing frame {frame_count}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        return "Error processing video", None
    finally:
        cap.release()
    
    # Clean up the input file if requested
    if cleanup and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"🗑️ Removed temporary file: {video_path}")
        except Exception as e:
            print(f"⚠️ Failed to remove temporary file {video_path}: {e}")
    
    if violation_detected:
        return f"Vehicle overspeeding violation detected: Vehicle ID {violating_vehicle_id} at {violating_vehicle_speed:.1f} km/h in {SPEED_LIMIT_KMH} km/h zone", violation_image_path
    else:
        return f"No vehicle overspeeding violations detected (processed {frame_count} frames)", None

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Process a video file to detect overspeeding violations.
    
    Args:
        file: The video file to process
    
    Returns:
        JSON response with the result and image path if violation detected
    """
    print(f"📹 Received video upload: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    # Generate unique filename for uploaded video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"
    
    # Save the uploaded file
    temp_file_path = os.path.join(current_dir, filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"💾 Video saved: {temp_file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")
    
    # Process the video synchronously
    print(f"🔍 Starting overspeeding detection...")
    start_time = time.time()
    
    result_message, image_path = process_video_overspeeding(
        temp_file_path,
        cleanup=True
    )
    
    processing_time = time.time() - start_time
    print(f"⏱️ Processing completed in {processing_time:.1f} seconds")
    
    # Prepare the response
    response_data = {
        "message": result_message,
        "processing_time": f"{processing_time:.1f}s",
        "speed_limit": f"{SPEED_LIMIT_KMH} km/h"
    }
    
    if image_path:
        # Get just the filename from the path
        image_filename = os.path.basename(image_path)
        response_data["violation_detected"] = True
        response_data["image_url"] = f"/images/{image_filename}"
        print(f"✅ Violation detected - image available at: /images/{image_filename}")
    else:
        response_data["violation_detected"] = False
        print(f"✅ No violations detected")
    
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
    """Root endpoint"""
    return {
        "message": "Overspeeding Detection API is running",
        "speed_limit": f"{SPEED_LIMIT_KMH} km/h",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global detector
    detector_status = "initialized" if detector is not None else "not_initialized"
    
    return {
        "status": "healthy",
        "detector": detector_status,
        "speed_limit": f"{SPEED_LIMIT_KMH} km/h",
        "outputs_dir": outputs_dir
    }

if __name__ == "__main__":
    # Initialize detector on startup
    initialize_detector()
    
    try:
        import uvicorn
        print("🚀 Starting Overspeeding Detection API on port 8007...")
        uvicorn.run(app, host="0.0.0.0", port=8007)
    except ImportError:
        print("❌ Uvicorn not installed. Install it with: pip install uvicorn")
