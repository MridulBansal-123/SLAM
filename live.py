import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import cv2
import tempfile
import os
import time

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ==================== MODEL DEFINITION ====================
# Must match the architecture used to train resnet152_depth_model.pth

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))


class ResNetDepthModel(nn.Module):
    def __init__(self):
        super(ResNetDepthModel, self).__init__()

        # ENCODER: Pre-trained ResNet-152
        original_model = models.resnet152(weights=None)

        # Extract layers for Skip Connections
        self.encoder_layer0 = nn.Sequential(original_model.conv1, original_model.bn1, original_model.relu)
        self.encoder_layer1 = nn.Sequential(original_model.maxpool, original_model.layer1)
        self.encoder_layer2 = original_model.layer2
        self.encoder_layer3 = original_model.layer3
        self.encoder_layer4 = original_model.layer4

        # DECODER: CNN with Upsampling
        self.up_block1 = UpSample(skip_input=2048 + 1024, output_features=1024)
        self.up_block2 = UpSample(skip_input=1024 + 512, output_features=512)
        self.up_block3 = UpSample(skip_input=512 + 256, output_features=256)
        self.up_block4 = UpSample(skip_input=256 + 64, output_features=128)

        # Final output layer (1 channel for Depth)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder Pass
        x0 = self.encoder_layer0(x)
        x1 = self.encoder_layer1(x0)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)

        # Decoder Pass (with Skips)
        d1 = self.up_block1(x4, x3)
        d2 = self.up_block2(d1, x2)
        d3 = self.up_block3(d2, x1)
        d4 = self.up_block4(d3, x0)

        return torch.sigmoid(self.final_conv(d4)) * 10.0


@st.cache_resource
def load_model():
    """Load the depth estimation model with GPU support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = ResNetDepthModel()
        # Load weights with map_location to handle CPU/GPU
        state_dict = torch.load("resnet152_depth_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        st.success(f"✅ Model loaded successfully on {device}")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device


def preprocess_image(image):
    """Preprocess image for the model."""
    original_size = image.size
    
    # Resize to model input size
    image = image.resize((640, 480))
    
    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, original_size


def predict_depth(model, device, input_tensor, original_size):
    """Run depth prediction."""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        depth_output = model(input_tensor)
        
        # Convert to numpy and resize back to original size
        depth_map = depth_output.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, original_size)
        
    return depth_map


def create_depth_frame(depth_map, colormap='magma'):
    """Create a colored depth frame for video (returns numpy array)"""
    # Normalize depth map to 0-1 range
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)
    
    # Convert to BGR for OpenCV (remove alpha channel and convert RGB to BGR)
    depth_bgr = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_bgr, cv2.COLOR_RGB2BGR)
    
    return depth_bgr


def process_video(model, device, video_path, colormap='magma', progress_bar=None, status_text=None):
    """Process video and generate depth estimation for each frame"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Failed to open video file"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output.name
    temp_output.close()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess and predict
        input_tensor, original_size = preprocess_image(pil_image)
        depth_map = predict_depth(model, device, input_tensor, original_size)
        
        # Create depth visualization frame
        depth_frame = create_depth_frame(depth_map, colormap)
        
        # Resize depth frame to match original video dimensions
        depth_frame_resized = cv2.resize(depth_frame, (frame_width, frame_height))
        
        # Write frame
        out.write(depth_frame_resized)
        
        frame_count += 1
        
        # Update progress
        if progress_bar is not None:
            progress_bar.progress(frame_count / total_frames)
        if status_text is not None:
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return temp_output_path, None


def process_video_side_by_side(model, device, video_path, colormap='magma', progress_bar=None, status_text=None):
    """Process video and generate side-by-side comparison (original + depth)"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Failed to open video file"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output.name
    temp_output.close()
    
    # Initialize video writer (double width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width * 2, frame_height))
    
    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess and predict
        input_tensor, original_size = preprocess_image(pil_image)
        depth_map = predict_depth(model, device, input_tensor, original_size)
        
        # Create depth visualization frame
        depth_frame = create_depth_frame(depth_map, colormap)
        
        # Resize depth frame to match original video dimensions
        depth_frame_resized = cv2.resize(depth_frame, (frame_width, frame_height))
        
        # Create side-by-side frame
        combined_frame = np.hstack([frame, depth_frame_resized])
        
        # Write frame
        out.write(combined_frame)
        
        frame_count += 1
        
        # Update progress
        if progress_bar is not None:
            progress_bar.progress(frame_count / total_frames)
        if status_text is not None:
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return temp_output_path, None


def main():
    st.set_page_config(
        page_title="Depth Estimation - ResNet152",
        page_icon="🔭",
        layout="wide"
    )
    
    st.title("🔭 Monocular Depth Estimation")
    st.markdown("### ResNet-152 Based Depth Estimation Model")
    
    # Sidebar
    st.sidebar.header("⚙ Settings")
    colormap = st.sidebar.selectbox(
        "Depth Colormap",
        ["magma", "plasma", "inferno", "viridis", "gray", "gray_r", "jet"],
        index=0
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model.")
        return
    
    st.sidebar.info(f"🖥 Device: {device}")
    
    # Quality settings in sidebar (shared by all live tabs)
    st.sidebar.subheader("📹 Live Stream Quality")
    resolution = st.sidebar.selectbox(
        "Resolution",
        ["360p (640x360)", "480p (854x480)", "720p (1280x720)", "Original"],
        index=0
    )
    
    # Parse resolution
    resolution_map = {
        "360p (640x360)": (640, 360),
        "480p (854x480)": (854, 480),
        "720p (1280x720)": (1280, 720),
        "Original": None
    }
    target_resolution = resolution_map[resolution]
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📷 Sample Images", "📤 Upload Image", "🎬 Upload Video", "💻 Laptop Webcam", "📱 Phone Camera", "ℹ About"])
    
    # ... (Keep Tab 1 and Tab 2 code exactly as they were) ...
    
    # ==================== VIDEO UPLOAD TAB ====================
    with tab3:
        st.header("🎬 Upload Video for Depth Estimation")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for depth estimation"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_input.write(uploaded_video.read())
            temp_input_path = temp_input.name
            temp_input.close()
            
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_video)
            
            # Get video info
            cap = cv2.VideoCapture(temp_input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("FPS", f"{fps}")
            with col3:
                st.metric("Frames", f"{total_frames}")
            with col4:
                st.metric("Resolution", f"{width}x{height}")
            
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
                
                with st.spinner("Processing video frames with GPU..."):
                    if output_mode == "Depth Only":
                        output_path, error = process_video(
                            model, device, temp_input_path, colormap,
                            progress_bar, status_text
                        )
                    else:
                        output_path, error = process_video_side_by_side(
                            model, device, temp_input_path, colormap,
                            progress_bar, status_text
                        )
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Video processing complete!")
                    status_text.text("Done!")
                    
                    # Display processed video
                    st.subheader("Depth Estimation Video")
                    
                    # Read the output video for display
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="depth_estimation_output.mp4",
                        mime="video/mp4"
                    )
                    
                    # Cleanup
                    os.unlink(output_path)
            
            # Cleanup input temp file
            try:
                os.unlink(temp_input_path)
            except:
                pass
    
    # ==================== LAPTOP WEBCAM TAB ====================
    with tab4:
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
        
        # Use button instead of toggle for better control
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
        
        # Placeholders for images
        col_web1, col_web2 = st.columns(2)
        with col_web1:
            st.subheader("Live Feed")
            webcam_frame_placeholder = st.empty()
        with col_web2:
            st.subheader("Live Depth")
            webcam_depth_placeholder = st.empty()
        
        # FPS counter
        fps_placeholder = st.empty()
        status_placeholder = st.empty()
            
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for faster Windows camera access
            
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
                    
                    # Resize frame to target resolution for faster processing
                    if target_resolution is not None:
                        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
                    
                    # 1. Process Frame for Display (OpenCV is BGR, PIL/Streamlit needs RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Display original
                    webcam_frame_placeholder.image(pil_image, use_container_width=True)
                    
                    # 2. Predict Depth
                    input_tensor, original_size = preprocess_image(pil_image)
                    depth_map = predict_depth(model, device, input_tensor, original_size)
                    
                    # 3. Create Visualization (faster method using numpy/cv2)
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                    depth_colored = plt.cm.get_cmap(colormap)(depth_normalized)
                    depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                    
                    # Display Depth
                    webcam_depth_placeholder.image(depth_rgb, use_container_width=True)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = frame_count / elapsed_time
                        fps_placeholder.metric("FPS", f"{fps:.1f}")
                    
                cap.release()
                status_placeholder.info("🔴 Webcam stopped.")
    
    # ==================== PHONE CAMERA TAB ====================
    with tab5:
        st.header("📱 Live Depth from Phone Camera")
        st.markdown("""
        1. Install *IP Webcam* (Android) or similar on your phone.
        2. Ensure phone and laptop are on the *same Wi-Fi*.
        3. Start the server on the phone and enter the IP address below (e.g., http://192.168.1.5:8080/video).
        """)
        
        # Input for the camera URL
        # Typical Android IP Webcam URL is http://<IP>:8080/video
        cam_url = st.text_input("Camera URL", "http://192.168.1.XX:8080/video")
        
        # Use button instead of toggle for better control
        col_pbtn1, col_pbtn2 = st.columns(2)
        with col_pbtn1:
            start_phone = st.button("▶️ Start Phone Camera", key="start_phone_btn")
        with col_pbtn2:
            stop_phone = st.button("⏹️ Stop Phone Camera", key="stop_phone_btn")
        
        # Session state for phone camera
        if 'phone_running' not in st.session_state:
            st.session_state.phone_running = False
        
        if start_phone:
            st.session_state.phone_running = True
        if stop_phone:
            st.session_state.phone_running = False
        
        # Placeholders for images
        col_live1, col_live2 = st.columns(2)
        with col_live1:
            st.subheader("Live Feed")
            frame_placeholder = st.empty()
        with col_live2:
            st.subheader("Live Depth")
            depth_placeholder = st.empty()
        
        # FPS counter
        phone_fps_placeholder = st.empty()
        phone_status_placeholder = st.empty()
            
        if st.session_state.phone_running:
            cap = cv2.VideoCapture(cam_url)
            
            if not cap.isOpened():
                st.error("Could not connect to phone camera. Check URL and Wi-Fi.")
                st.session_state.phone_running = False
            else:
                phone_status_placeholder.success("🟢 Phone camera is running... Click 'Stop Phone Camera' to end.")
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.phone_running:
                    ret, frame = cap.read()
                    if not ret:
                        phone_status_placeholder.warning("Frame drop or stream ended.")
                        break
                    
                    # Resize frame to target resolution for faster processing
                    if target_resolution is not None:
                        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
                    
                    # 1. Process Frame for Display (OpenCV is BGR, PIL/Streamlit needs RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Display original
                    frame_placeholder.image(pil_image, use_container_width=True)
                    
                    # 2. Predict Depth
                    input_tensor, original_size = preprocess_image(pil_image)
                    depth_map = predict_depth(model, device, input_tensor, original_size)
                    
                    # 3. Create Visualization (faster method)
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                    depth_colored = plt.cm.get_cmap(colormap)(depth_normalized)
                    depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                    
                    # Display Depth
                    depth_placeholder.image(depth_rgb, use_container_width=True)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = frame_count / elapsed_time
                        phone_fps_placeholder.metric("FPS", f"{fps:.1f}")
                    
                cap.release()
                phone_status_placeholder.info("🔴 Phone camera stopped.")

    # ==================== ABOUT TAB ====================
    with tab6:
        st.header("About This Application")
        st.markdown("""
        ### Model Architecture
        This depth estimation model uses a **ResNet-152** backbone as an encoder with a custom decoder featuring skip connections.
        
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
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and PyTorch")


if __name__ == "__main__":
    main()