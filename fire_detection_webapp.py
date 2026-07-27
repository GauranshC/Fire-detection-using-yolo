# fire_detection_webapp.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import cvzone
import math

# Set page config
st.set_page_config(
    page_title="🔥 Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load and cache the YOLO model"""
    try:
        model = YOLO('fire.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_fire_in_frame(model, frame):
    """Detect fire in frame"""
    if model is None:
        return frame, []
    
    results = model(frame, stream=True)
    detections = []
    
    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence_percent = math.ceil(confidence * 100)
            class_id = int(box.cls[0])
            
            if confidence_percent > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(frame, f'Fire {confidence_percent}%', 
                                 [x1 + 8, y1 + 100], scale=1, thickness=2)
                
                detections.append({
                    'confidence': confidence_percent,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return frame, detections

def main():
    
    st.title("🔥 Fire Detection System")
    st.markdown("*Choose your input source for real-time fire detection*")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Could not load fire detection model!")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 10, 100, 50, 5)
        
        st.header("📊 Statistics") 
        if 'detection_count' not in st.session_state:
            st.session_state.detection_count = 0
        
        st.metric("🔥 Total Detections", st.session_state.detection_count)
        
        if st.button("🔄 Reset Counter"):
            st.session_state.detection_count = 0
            st.success("Counter reset!")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📹 Live Camera"])
    
    with tab1:
        st.subheader("📷 Image Fire Detection")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect fire"
        )
        
        if uploaded_image is not None:
            # Display original image
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))
            
            with st.spinner("🔍 Detecting fire..."):
                annotated_frame, detections = detect_fire_in_frame(model, frame)
            
            with col2:
                st.subheader("Detection Result")
                result_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(result_image, use_column_width=True)
            
            # Results
            if detections:
                st.error(f"🔥 **FIRE DETECTED!** Found {len(detections)} detection(s)")
                st.session_state.detection_count += len(detections)
                
                # Show details
                for i, detection in enumerate(detections):
                    st.write(f"🔥 Detection {i+1}: {detection['confidence']}% confidence")
            else:
                st.success("✅ No fire detected in the image")
    with tab2:
        st.subheader("🎥 Video Fire Detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect fire"
        )
        
        if uploaded_video is not None:
            # Save video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            # Display video info
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            st.info(f"📹 Video: {frame_count} frames, {fps} FPS, {duration:.1f}s duration")
            
            if st.button("🚀 Start Processing"):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                video_placeholder = st.empty()
                
                detections_found = []
                processed_frames = 0
                
                cap = cv2.VideoCapture(tfile.name)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frames += 1
                    
                    # Process every 10th frame for speed
                    if processed_frames % 10 == 0:
                        frame = cv2.resize(frame, (640, 480))
                        annotated_frame, detections = detect_fire_in_frame(model, frame)
                        
                        if detections:
                            detections_found.extend(detections)
                            st.session_state.detection_count += len(detections)
                        
                        # Update display
                        result_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(result_image, use_column_width=True)
                    
                    # Update progress
                    progress = processed_frames / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {processed_frames}/{frame_count}")
                
                cap.release()
                
                # Final results
                if detections_found:
                    st.error(f"🔥 **FIRE DETECTED!** Found {len(detections_found)} total detection(s)")
                else:
                    st.success("✅ No fire detected in the video")
            
            # Cleanup
            try:
                os.unlink(tfile.name)
            except:
                pass
    
    with tab3:
        st.subheader("📹 Live Camera Detection")
        
        st.info("🎥 **Camera Access Required**")
        st.markdown("""
        **Setup Instructions:**
        1. Make sure your camera is connected
        2. Grant camera permissions to your browser
        3. Click 'Start Camera' below
        """)
        
        # Camera controls
        col1, col2 = st.columns(2)
        with col1:
            camera_index = st.selectbox("📷 Select Camera", [0, 1, 2], help="Try different indices if camera not found")
        with col2:
            detection_sensitivity = st.slider("🔍 Detection Sensitivity", 1, 10, 5)
        
        if st.button("🎬 Start Camera"):
            # Camera detection
            camera_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    st.error(f"❌ Could not open camera {camera_index}")
                else:
                    st.success("✅ Camera connected successfully!")
                    
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Live detection loop
                    frame_counter = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_counter += 1
                        
                        # Process every few frames
                        if frame_counter % 5 == 0:
                            annotated_frame, detections = detect_fire_in_frame(model, frame)
                            
                            if detections:
                                status_placeholder.error(f"🚨 FIRE ALERT! {len(detections)} detection(s)")
                                st.session_state.detection_count += len(detections)
                            else:
                                status_placeholder.success("✅ Monitoring... All clear")
                            
                            # Display frame
                            result_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(result_image, use_column_width=True)
                        
                        # Break condition (in real app, you'd have a stop button)
                        if frame_counter > 1000:  # Limit for demo
                            break
                    
                    cap.release()
                    
            except Exception as e:
                st.error(f"❌ Camera error: {e}")
                st.markdown("""
                **Troubleshooting:**
                - Check camera permissions in browser settings
                - Try different camera index (0, 1, 2)
                - Restart browser if needed
                """)

if __name__ == "__main__":
    main()
