from speed_detector import SpeedDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description='Detect overspeeding vehicles in dashcam footage')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize and run detector
    detector = SpeedDetector(args.video_path, args.output_dir)
    detector.process_video()
    
if __name__ == "__main__":
    main()