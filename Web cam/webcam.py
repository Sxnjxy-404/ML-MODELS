#!/usr/bin/env python3
"""
YOLOv8 Live Webcam Detection (OpenCV window)
-------------------------------------------------------------
Installation:
pip install ultralytics opencv-python

Usage:
python webcam.py --model yolov8n.pt --device 0
Press 'q' to quit.
"""

import argparse
import cv2
from ultralytics import YOLO
import time


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="YOLOv8 Live Webcam Detection")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="Path to YOLOv8 *.pt model")
    p.add_argument("--device", type=str, default="", 
                   help="CUDA device like '0' or '0,1' or 'cpu' (auto if empty)")
    p.add_argument("--source", type=int, default=0,
                   help="Webcam index (default 0)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--show-fps", action="store_true",
                   help="Overlay FPS counter")
    p.add_argument("--img-size", type=int, default=640,
                   help="Inference image size")
    return p.parse_args()


def main():
    """Main function to run YOLOv8 webcam detection."""
    args = parse_args()
    
    # Load YOLOv8 model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Initialize webcam with different backends
    print(f"Opening webcam index: {args.source}")
    
    # Try different backends for Windows compatibility
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        print(f"Trying backend: {backend}")
        cap = cv2.VideoCapture(args.source, backend)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"Successfully opened webcam with backend: {backend}")
                break
            else:
                cap.release()
                cap = None
        else:
            if cap:
                cap.release()
                cap = None
    
    if cap is None or not cap.isOpened():
        print("Available cameras:")
        for i in range(10):  # Check first 10 camera indices
            test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    print(f"  Camera {i}: Available")
                test_cap.release()
        raise RuntimeError(f"Cannot open any webcam. Try different --source values (0, 1, 2, etc.)")
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    
    # FPS calculation variables
    prev_time = time.time()
    fps = 0.0
    
    print("Starting detection... Press 'q' to quit")
    frame_count = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            frame_count += 1
            if frame_count == 1:
                print("First frame captured successfully!")
                print(f"Frame size: {frame.shape}")
            
            # Run YOLOv8 inference
            if frame_count == 1:
                print("Running first inference (may take a moment)...")
            
            results = model.predict(
                source=frame, 
                conf=args.conf,
                verbose=False, 
                device=args.device,
                imgsz=args.img_size
            )
            
            if frame_count == 1:
                print("First inference completed! Opening display window...")
            
            # Draw annotations on frame
            annotated_frame = results[0].plot()  # BGR image with boxes & labels
            
            # Calculate and display FPS if requested
            if args.show_fps:
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                
                if dt > 0:
                    # Smooth FPS calculation using exponential moving average
                    current_fps = 1.0 / dt
                    fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
                
                # Draw FPS on frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
            
            # Check for 'q' key press to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


if __name__ == "__main__":
    main()