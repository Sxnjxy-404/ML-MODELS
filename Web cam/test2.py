import cv2

def test_cameras():
    """Test which cameras are available."""
    print("Testing camera access...")
    
    working_cameras = []
    
    for i in range(10):  # Test first 10 indices
        print(f"\nTesting camera {i}...")
        
        # Try DirectShow first (recommended for Windows)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Camera {i}: Working! Resolution: {w}x{h}")
                working_cameras.append(i)
                
                # Show frame for 2 seconds
                cv2.imshow(f"Camera {i} Test", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                print(f"✗ Camera {i}: Opens but can't read frames")
        else:
            print(f"✗ Camera {i}: Cannot open")
        
        cap.release()
    
    if working_cameras:
        print(f"\n✓ Working cameras found: {working_cameras}")
        print(f"Use --source {working_cameras[0]} in your main script")
    else:
        print("\n✗ No working cameras found!")
        print("\nTroubleshooting:")
        print("1. Check Windows camera privacy settings")
        print("2. Close other apps using the camera")
        print("3. Try running as Administrator")
        print("4. Restart your computer")

if __name__ == "__main__":
    test_cameras()