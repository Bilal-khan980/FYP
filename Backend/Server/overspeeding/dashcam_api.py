"""
FastAPI endpoint for dashcam overspeeding detection
Follows the FYP project pattern for consistency with other modules
"""

import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dashcam_speed_detector import DashcamSpeedDetector


# Initialize FastAPI app
app = FastAPI(title="Dashcam Overspeeding Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Initialize the speed detector
try:
    pwc_model_path = os.path.join(current_dir, "pwc_net.pth")
    detector = DashcamSpeedDetector(
        pwc_model_path=pwc_model_path if os.path.exists(pwc_model_path) else None,
        yolo_model_path="yolov8n.pt",
        speed_model_type="nvidia",
        device="auto"
    )
    print("Dashcam speed detector initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize speed detector: {e}")
    detector = None


def process_video_overspeeding(video_path: str, cleanup: bool = True) -> tuple[str, Optional[str]]:
    """
    Process video for overspeeding detection
    
    Args:
        video_path: Path to the input video
        cleanup: Whether to remove the input video after processing
        
    Returns:
        Tuple containing (result_message, image_path) where:
        - result_message: "Overspeeding violation detected" or "No violation"
        - image_path: Path to the saved image if violation detected, None otherwise
    """
    if detector is None:
        return "Error: Speed detector not initialized", None
    
    try:
        # Generate output video path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(outputs_dir, f"overspeeding_{timestamp}.mp4")
        
        # Process the video
        results = detector.process_video(
            video_path=video_path,
            output_path=output_video_path,
            max_frames=1000  # Limit processing for API responsiveness
        )
        
        # Check if violations were detected
        violation_detected = len(results['violation_frames']) > 0
        violation_image_path = None
        
        if violation_detected:
            # Extract a frame from the violation for display
            violation_frame_idx = results['violation_frames'][0]['frame']
            violation_image_path = extract_violation_frame(
                output_video_path, 
                violation_frame_idx, 
                timestamp
            )
            
            result_message = f"Overspeeding violation detected! Max speed: {results['max_ego_speed']:.1f} km/h"
        else:
            result_message = f"No overspeeding violation detected. Max speed: {results['max_ego_speed']:.1f} km/h"
        
        # Cleanup input video if requested
        if cleanup and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Warning: Could not remove input video: {e}")
        
        return result_message, violation_image_path
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return f"Error processing video: {str(e)}", None


def extract_violation_frame(video_path: str, frame_idx: int, timestamp: str) -> str:
    """
    Extract a specific frame from video for violation evidence
    
    Args:
        video_path: Path to the processed video
        frame_idx: Frame index to extract
        timestamp: Timestamp for unique filename
        
    Returns:
        Path to the extracted frame image
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Seek to the violation frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Save the frame as an image
            image_filename = f"violation_{timestamp}_frame{frame_idx}.jpg"
            image_path = os.path.join(outputs_dir, image_filename)
            cv2.imwrite(image_path, frame)
            cap.release()
            return image_path
        else:
            cap.release()
            return None
            
    except Exception as e:
        print(f"Error extracting violation frame: {e}")
        return None


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Process a dashcam video file to detect overspeeding violations.
    
    Args:
        file: The video file to process
        
    Returns:
        JSON response with the result and image path if violation detected
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Speed detector not initialized")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate a unique filename for the uploaded video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"
    
    # Save the uploaded file
    temp_file_path = os.path.join(current_dir, filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")
    
    # Process the video synchronously
    result_message, image_path = process_video_overspeeding(
        temp_file_path,
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


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """
    Get a processed video file.
    
    Args:
        filename: The name of the video file
        
    Returns:
        The video file
    """
    file_path = os.path.join(outputs_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dashcam Overspeeding Detection API is running",
        "version": "1.0.0",
        "detector_status": "initialized" if detector else "error"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_initialized": detector is not None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8007)
    except ImportError:
        print("Uvicorn not installed. Install it with: pip install uvicorn")
    except Exception as e:
        print(f"Error starting server: {e}")
