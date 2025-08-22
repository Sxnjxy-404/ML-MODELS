import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import logging

# Configure logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Live Detection", 
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_model(model_path: str):
    """Load and cache YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

# Main UI
st.title("🎯 YOLOv8 Live Object Detection")
st.write("Real-time object detection using your webcam. Press **Stop** to end the stream.")

# Create columns for controls
col1, col2, col3 = st.columns(3)

with col1:
    model_name = st.selectbox(
        "Select Model", 
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], 
        index=0,
        help="Larger models are more accurate but slower"
    )

with col2:
    conf = st.slider(
        "Confidence Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=0.25, 
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )

with col3:
    device = st.selectbox(
        "Device", 
        ["auto", "cpu", "cuda"], 
        index=0,
        help="Use CUDA for GPU acceleration if available"
    )

# Additional options
with st.expander("Advanced Options", expanded=False):
    classes_text = st.text_input(
        "Filter Classes (optional)", 
        value="",
        placeholder="person, car, dog (comma-separated)",
        help="Leave empty to detect all classes, or specify class names to filter"
    )
    
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)

# Load model
model = load_model(model_name)

if model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Process class filtering
CLASS_NAME_TO_ID = None
if classes_text.strip():
    try:
        # Build name->id mapping from model.names
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        selected_classes = [s.strip().lower() for s in classes_text.split(",") if s.strip()]
        valid_classes = [s for s in selected_classes if s in name_to_id]
        
        if valid_classes:
            CLASS_NAME_TO_ID = [name_to_id[s] for s in valid_classes]
            st.info(f"Filtering for classes: {', '.join(valid_classes)}")
        else:
            st.warning("No valid class names found. Detecting all classes.")
            
        if len(valid_classes) < len(selected_classes):
            invalid_classes = [s for s in selected_classes if s not in name_to_id]
            st.warning(f"Invalid class names ignored: {', '.join(invalid_classes)}")
            
    except Exception as e:
        st.error(f"Error processing class filter: {str(e)}")

# WebRTC configuration
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Run YOLO prediction
            results = model.predict(
                img, 
                conf=conf, 
                verbose=False,
                device=(None if device == "auto" else device),
                classes=CLASS_NAME_TO_ID
            )
            
            # Get annotated frame
            if results and len(results) > 0:
                # Customize plot parameters
                annotated = results[0].plot(
                    labels=show_labels,
                    conf=show_conf
                )
            else:
                annotated = img
                
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            # Return original frame on error
            st.error(f"Processing error: {str(e)}")
            return frame

# Display available classes
with st.expander("Available Classes", expanded=False):
    if model and hasattr(model, 'names'):
        classes_list = list(model.names.values())
        st.write(f"Total classes: {len(classes_list)}")
        
        # Display classes in columns
        cols = st.columns(4)
        for i, class_name in enumerate(classes_list):
            with cols[i % 4]:
                st.write(f"• {class_name}")

# WebRTC streamer
st.markdown("---")
st.subheader("Live Detection")

try:
    webrtc_streamer(
        key="yolov8-live-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480}, 
            "audio": False
        },
        async_processing=True,
        rtc_configuration=rtc_config,
    )
except Exception as e:
    st.error(f"WebRTC streaming error: {str(e)}")
    st.info("Please make sure you have a working webcam and have granted camera permissions.")

# Instructions
with st.expander("Instructions", expanded=False):
    st.markdown("""
    1. **Select a model**: Smaller models (n, s) are faster, larger models (l, x) are more accurate
    2. **Adjust confidence**: Lower values detect more objects, higher values are more selective
    3. **Choose device**: Use 'cuda' if you have a compatible GPU for faster processing
    4. **Filter classes**: Optionally specify which object types to detect
    5. **Click 'Start'** to begin detection
    6. **Allow camera access** when prompted by your browser
    7. **Click 'Stop'** to end the stream
    
    **Note**: This runs entirely in your browser. No video data is sent to external servers.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, YOLOv8, and WebRTC")