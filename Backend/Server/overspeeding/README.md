# Dashcam Overspeeding Detection System

A modern AI-powered system for detecting overspeeding violations from dashcam videos using optical flow and deep learning techniques.

## Features

- **PWC-Net Optical Flow**: Advanced neural network-based optical flow estimation
- **NVIDIA CNN Architecture**: Speed regression using proven CNN architecture from NVIDIA's self-driving car research
- **YOLOv8 Vehicle Detection**: State-of-the-art object detection for vehicle tracking
- **Real-time Processing**: Optimized for real-time video processing
- **FastAPI Integration**: RESTful API following the FYP project pattern
- **Violation Detection**: Automatic detection and logging of overspeeding violations

## System Architecture

```
Input Video → Frame Extraction → Optical Flow (PWC-Net) → Speed Estimation (NVIDIA CNN)
                ↓
Vehicle Detection (YOLOv8) → Vehicle Tracking → Violation Detection → Output Video
```

## Installation

1. **Install Dependencies**:
   ```bash
   cd /home/bilalkhan/Final_Year_Project/Overspeeding
   pip install -r requirements.txt
   ```

2. **Download Models** (if not already present):
   - YOLOv8 model will be downloaded automatically
   - PWC-Net model (`pwc_net.pth`) should be in the Overspeeding folder

## Usage

### 1. Command Line Interface

**Create a test video**:
```bash
python test_video.py --create-video --duration 10
```

**Test speed detection**:
```bash
python test_video.py --test-detection --video-path test_video.mp4
```

**Process your own video**:
```python
from dashcam_speed_detector import DashcamSpeedDetector

detector = DashcamSpeedDetector()
results = detector.process_video("your_video.mp4", "output_video.mp4")
print(f"Max speed detected: {results['max_ego_speed']:.1f} km/h")
```

### 2. FastAPI Server

**Start the API server**:
```bash
python dashcam_api.py
```

The server will start on `http://localhost:8007`

**API Endpoints**:
- `POST /process-video/` - Upload and process a video
- `GET /images/{filename}` - Get violation evidence images
- `GET /videos/{filename}` - Get processed videos
- `GET /health` - Health check

**Test the API**:
```bash
curl -X POST 'http://localhost:8007/process-video/' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@test_video.mp4'
```

### 3. Integration with FYP Frontend

The API follows the same pattern as other FYP modules:

1. **Upload**: Videos are uploaded via the frontend
2. **Processing**: The system processes videos and detects violations
3. **Results**: Violation frames are saved and returned to the frontend
4. **Display**: Results are displayed in the FYP results interface

## Technical Details

### PWC-Net Optical Flow
- Modern neural network approach to optical flow estimation
- Pre-trained on MPI Sintel dataset
- Provides dense optical flow fields for accurate motion estimation

### NVIDIA CNN Architecture
- Based on End-to-End Deep Learning for Self-Driving Cars
- Convolutional layers: 24, 36, 48, 64, 64 filters
- Fully connected layers: 100, 50, 10, 1 neurons
- Batch normalization and dropout for regularization

### Speed Calculation
1. **Optical Flow**: Calculate dense optical flow between consecutive frames
2. **Ego Motion**: Estimate camera/vehicle movement from flow
3. **Speed Regression**: Use CNN to predict speed from optical flow
4. **Calibration**: Apply calibration factors for real-world speed

### Vehicle Detection and Tracking
- YOLOv8 for real-time vehicle detection
- Simple tracking based on position proximity
- Speed estimation for detected vehicles relative to ego motion

## Configuration

### Speed Detection Parameters
```python
detector = DashcamSpeedDetector(
    pwc_model_path="pwc_net.pth",      # PWC-Net model path
    yolo_model_path="yolov8n.pt",      # YOLO model path
    speed_model_type="nvidia",          # CNN architecture type
    device="auto"                       # Device (auto/cuda/cpu)
)

# Adjust detection parameters
detector.speed_limit = 120              # Speed limit in km/h
detector.confidence_threshold = 0.5     # YOLO confidence threshold
detector.speed_scale = 0.02            # Speed calibration factor
```

## File Structure

```
Overspeeding/
├── dashcam_speed_detector.py    # Main detection class
├── pwc_net.py                   # PWC-Net implementation
├── nvidia_cnn.py                # NVIDIA CNN architecture
├── optical_flow_utils.py        # Optical flow utilities
├── dashcam_api.py              # FastAPI server
├── test_video.py               # Testing utilities
├── requirements.txt            # Dependencies
├── pwc_net.pth                # Pre-trained PWC-Net model
├── outputs/                    # Output videos and images
└── README.md                   # This file
```

## Performance

- **Processing Speed**: ~15-30 FPS on GPU, ~5-10 FPS on CPU
- **Accuracy**: Depends on calibration and video quality
- **Memory Usage**: ~2-4 GB GPU memory for processing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size or use CPU processing
   - Set `device="cpu"` in detector initialization

2. **Model Loading Errors**:
   - Ensure `pwc_net.pth` is in the correct directory
   - Check PyTorch version compatibility

3. **Slow Processing**:
   - Use GPU if available
   - Reduce video resolution or frame rate
   - Limit `max_frames` parameter

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Process with limited frames for debugging
results = detector.process_video("video.mp4", max_frames=100)
```

## Integration with FYP System

This module integrates seamlessly with the existing FYP traffic violation detection system:

1. **Backend Integration**: Add to `server.js` processing pipeline
2. **Frontend Integration**: Use existing upload and results components
3. **Database Integration**: Store results in MongoDB following existing schema
4. **API Consistency**: Follows same endpoint patterns as other modules

## Future Improvements

- **Calibration**: Implement automatic camera calibration
- **Multi-object Tracking**: Advanced tracking algorithms (DeepSORT)
- **Speed Accuracy**: Improve speed estimation with depth information
- **Real-time Streaming**: Support for live video streams
- **Mobile Optimization**: Optimize for mobile deployment

## References

- PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
- NVIDIA End-to-End Deep Learning for Self-Driving Cars
- YOLOv8: Ultralytics Object Detection
- Original Voof implementation by antoninodimaggio
