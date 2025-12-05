"""
SLAM - Monocular Depth Estimation Web Application

A Streamlit-based web application for real-time monocular depth estimation
using a ResNet-152 encoder-decoder architecture.

Features:
    - Upload and process images
    - Upload and process videos
    - Real-time depth estimation from laptop webcam
    - Real-time depth estimation from phone camera (via IP Webcam)

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.model import ResNetDepthModel
from src.inference import DepthEstimator, colorize_depth
from src.video import VideoProcessor
from src.utils import print_device_info


# Print device info on startup
print_device_info()


# ==================== CACHED MODEL LOADING ====================

@st.cache_resource
def load_model() -> Tuple[Optional[DepthEstimator], str]:
    """
    Load and cache the depth estimation model.
    
    Returns:
        Tuple of (DepthEstimator instance, device string)
    """
    estimator = DepthEstimator()
    
    if estimator.load_model():
        st.success(f"✅ Model loaded successfully on {estimator.device}")
        return estimator, str(estimator.device)
    else:
        st.error(f"❌ Failed to load model from {estimator.model_path}")
        return None, str(estimator.device)


# ==================== TAB IMPLEMENTATIONS ====================

def render_sample_images_tab(estimator: DepthEstimator, colormap: str) -> None:
    """Render the sample images tab content."""
    st.header("📷 Sample Images")
    st.markdown("Test the depth estimation model with sample images from the repository.")
    
    # Look for sample images
    sample_dir = os.path.dirname(os.path.abspath(__file__))
    sample_images = [f for f in os.listdir(sample_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sample_images:
        st.info("No sample images found. Add .jpg or .png files to the SLAM directory.")
        return
    
    selected_image = st.selectbox("Select a sample image", sample_images)
    
    if st.button("🔍 Analyze Sample", key="btn_sample"):
        image_path = os.path.join(sample_dir, selected_image)
        image = Image.open(image_path).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Depth Map")
            with st.spinner("Predicting depth..."):
                depth_map = estimator.estimate_depth(image)
                depth_colored = colorize_depth(depth_map, colormap)
            st.image(depth_colored, use_container_width=True)


def render_upload_image_tab(estimator: DepthEstimator, colormap: str) -> None:
    """Render the upload image tab content."""
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for depth estimation"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Depth Map")
            with st.spinner("Predicting depth..."):
                depth_map = estimator.estimate_depth(image)
                depth_colored = colorize_depth(depth_map, colormap)
            st.image(depth_colored, use_container_width=True)
        
        # Display depth statistics
        st.markdown("### Depth Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Depth", f"{depth_map.min():.2f}m")
        with col2:
            st.metric("Max Depth", f"{depth_map.max():.2f}m")
        with col3:
            st.metric("Mean Depth", f"{depth_map.mean():.2f}m")


def render_upload_video_tab(
    estimator: DepthEstimator, 
    colormap: str
) -> None:
    """Render the upload video tab content."""
    st.header("🎬 Upload Video for Depth Estimation")
    
    uploaded_video = st.file_uploader(
        "Choose a video file...",
        type=config.SUPPORTED_VIDEO_FORMATS,
        help="Upload a video file for depth estimation"
    )
    
    if uploaded_video is None:
        return
    
    # Save uploaded video to temporary file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_video.read())
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Display original video
    st.subheader("Original Video")
    st.video(uploaded_video)
    
    # Get and display video info
    video_processor = VideoProcessor(estimator)
    video_info = video_processor.get_video_info(temp_input_path)
    
    if video_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{video_info['duration']:.1f}s")
        with col2:
            st.metric("FPS", f"{video_info['fps']}")
        with col3:
            st.metric("Frames", f"{video_info['total_frames']}")
        with col4:
            st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
    
    # Processing options
    st.subheader("Processing Options")
    output_mode = st.radio(
        "Output Mode",
        ["Depth Only", "Side-by-Side (Original + Depth)"],
        horizontal=True
    )
    
    if st.button("🚀 Process Video", key="btn_video"):
        st.subheader("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing video frames..."):
            if output_mode == "Depth Only":
                output_path, error = video_processor.process_video(
                    temp_input_path, colormap,
                    progress_callback=progress_bar.progress,
                    status_callback=status_text.text
                )
            else:
                output_path, error = video_processor.process_video_side_by_side(
                    temp_input_path, colormap,
                    progress_callback=progress_bar.progress,
                    status_callback=status_text.text
                )
        
        if error:
            st.error(f"❌ Error: {error}")
        else:
            st.success("✅ Video processing complete!")
            status_text.text("Done!")
            
            # Display and download processed video
            st.subheader("Depth Estimation Video")
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            
            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name="depth_estimation_output.mp4",
                mime="video/mp4"
            )
            
            os.unlink(output_path)
    
    # Cleanup input temp file
    try:
        os.unlink(temp_input_path)
    except:
        pass


def render_webcam_tab(
    estimator: DepthEstimator, 
    colormap: str,
    target_resolution: Optional[Tuple[int, int]]
) -> None:
    """Render the laptop webcam tab content."""
    st.header("💻 Live Depth from Laptop Webcam")
    st.markdown("Use your laptop's built-in webcam for real-time depth estimation.")
    
    # Camera selection
    camera_index = st.selectbox(
        "Select Camera",
        [0, 1, 2],
        format_func=lambda x: f"Camera {x}" + (" (Default)" if x == 0 else ""),
        index=0,
        key="webcam_camera_select"
    )
    
    # Control buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_webcam = st.button("▶️ Start Webcam", key="start_webcam_btn")
    with col_btn2:
        stop_webcam = st.button("⏹️ Stop Webcam", key="stop_webcam_btn")
    
    # Session state for webcam
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_webcam:
        st.session_state.webcam_running = True
    if stop_webcam:
        st.session_state.webcam_running = False
    
    # Placeholders
    col_web1, col_web2 = st.columns(2)
    with col_web1:
        st.subheader("Live Feed")
        webcam_frame_placeholder = st.empty()
    with col_web2:
        st.subheader("Live Depth")
        webcam_depth_placeholder = st.empty()
    
    fps_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.session_state.webcam_running:
        # CAP_DSHOW for faster Windows camera access
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_index}. Try a different camera index.")
            st.session_state.webcam_running = False
        else:
            status_placeholder.success("🟢 Webcam is running... Click 'Stop Webcam' to end.")
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.warning("Failed to grab frame. Retrying...")
                    continue
                
                # Resize frame for faster processing
                if target_resolution is not None:
                    frame = cv2.resize(
                        frame, target_resolution, 
                        interpolation=cv2.INTER_AREA
                    )
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                webcam_frame_placeholder.image(pil_image, use_container_width=True)
                
                # Predict depth
                depth_map = estimator.estimate_depth(pil_image)
                depth_colored = colorize_depth(depth_map, colormap)
                
                webcam_depth_placeholder.image(depth_colored, use_container_width=True)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            cap.release()
            status_placeholder.info("🔴 Webcam stopped.")


def render_phone_camera_tab(
    estimator: DepthEstimator, 
    colormap: str,
    target_resolution: Optional[Tuple[int, int]]
) -> None:
    """Render the phone camera tab content."""
    st.header("📱 Live Depth from Phone Camera")
    st.markdown("""
    1. Install **IP Webcam** (Android) or similar on your phone.
    2. Ensure phone and laptop are on the **same Wi-Fi**.
    3. Start the server on the phone and enter the IP address below.
    """)
    
    cam_url = st.text_input(
        "Camera URL", 
        "http://192.168.1.XX:8080/video",
        help="Example: http://192.168.1.100:8080/video"
    )
    
    # Control buttons
    col_pbtn1, col_pbtn2 = st.columns(2)
    with col_pbtn1:
        start_phone = st.button("▶️ Start Phone Camera", key="start_phone_btn")
    with col_pbtn2:
        stop_phone = st.button("⏹️ Stop Phone Camera", key="stop_phone_btn")
    
    # Session state
    if 'phone_running' not in st.session_state:
        st.session_state.phone_running = False
    
    if start_phone:
        st.session_state.phone_running = True
    if stop_phone:
        st.session_state.phone_running = False
    
    # Placeholders
    col_live1, col_live2 = st.columns(2)
    with col_live1:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()
    with col_live2:
        st.subheader("Live Depth")
        depth_placeholder = st.empty()
    
    phone_fps_placeholder = st.empty()
    phone_status_placeholder = st.empty()
    
    if st.session_state.phone_running:
        cap = cv2.VideoCapture(cam_url)
        
        if not cap.isOpened():
            st.error("Could not connect to phone camera. Check URL and Wi-Fi.")
            st.session_state.phone_running = False
        else:
            phone_status_placeholder.success(
                "🟢 Phone camera is running... Click 'Stop Phone Camera' to end."
            )
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.phone_running:
                ret, frame = cap.read()
                if not ret:
                    phone_status_placeholder.warning("Frame drop or stream ended.")
                    break
                
                # Resize frame for faster processing
                if target_resolution is not None:
                    frame = cv2.resize(
                        frame, target_resolution, 
                        interpolation=cv2.INTER_AREA
                    )
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frame_placeholder.image(pil_image, use_container_width=True)
                
                # Predict depth
                depth_map = estimator.estimate_depth(pil_image)
                depth_colored = colorize_depth(depth_map, colormap)
                
                depth_placeholder.image(depth_colored, use_container_width=True)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    phone_fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            cap.release()
            phone_status_placeholder.info("🔴 Phone camera stopped.")


def render_about_tab() -> None:
    """Render the about tab content."""
    st.header("About This Application")
    st.markdown("""
    ### Model Architecture
    This depth estimation model uses a **ResNet-152** backbone as an encoder 
    with a custom decoder featuring skip connections.
    
    **Key Features:**
    - **Encoder**: ResNet-152 for feature extraction
    - **Decoder**: Custom upsampling blocks with skip connections
    - **Output**: Depth map scaled to 0-10 meters
    
    ### Available Features
    - **Sample Images**: Test with pre-loaded images
    - **Upload Image**: Process your own images
    - **Upload Video**: Process video files with depth estimation
    - **Laptop Webcam**: Real-time depth from built-in camera
    - **Phone Camera**: Stream from IP Webcam app
    
    ### Performance Tips
    - Use **360p** resolution for fastest processing
    - GPU acceleration is enabled when CUDA is available
    - Lower resolution = Higher FPS
    - For videos, processing time depends on frame count
    
    ### Technology Stack
    - **PyTorch**: Deep learning framework
    - **Streamlit**: Web application framework
    - **OpenCV**: Computer vision library
    - **ResNet-152**: Backbone architecture
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch")


# ==================== MAIN APPLICATION ====================

def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.PAGE_LAYOUT
    )
    
    # Header
    st.title("🔭 Monocular Depth Estimation")
    st.markdown("### ResNet-152 Based Depth Estimation Model")
    
    # Sidebar - Settings
    st.sidebar.header("⚙️ Settings")
    colormap = st.sidebar.selectbox(
        "Depth Colormap",
        config.COLORMAPS,
        index=config.COLORMAPS.index(config.DEFAULT_COLORMAP)
    )
    
    # Load model
    with st.spinner("Loading model..."):
        estimator, device = load_model()
    
    if estimator is None:
        st.error("Failed to load the model. Please check the model path.")
        return
    
    st.sidebar.info(f"🖥️ Device: {device}")
    
    # Sidebar - Live Stream Quality
    st.sidebar.subheader("📹 Live Stream Quality")
    resolution = st.sidebar.selectbox(
        "Resolution",
        list(config.RESOLUTION_PRESETS.keys()),
        index=list(config.RESOLUTION_PRESETS.keys()).index(config.DEFAULT_RESOLUTION)
    )
    target_resolution = config.RESOLUTION_PRESETS[resolution]
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📷 Sample Images",
        "📤 Upload Image", 
        "🎬 Upload Video",
        "💻 Laptop Webcam",
        "📱 Phone Camera",
        "ℹ️ About"
    ])
    
    with tab1:
        render_sample_images_tab(estimator, colormap)
    
    with tab2:
        render_upload_image_tab(estimator, colormap)
    
    with tab3:
        render_upload_video_tab(estimator, colormap)
    
    with tab4:
        render_webcam_tab(estimator, colormap, target_resolution)
    
    with tab5:
        render_phone_camera_tab(estimator, colormap, target_resolution)
    
    with tab6:
        render_about_tab()


if __name__ == "__main__":
    main()
