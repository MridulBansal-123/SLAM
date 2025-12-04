import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 Sample Images", "📤 Upload Image", "💻 Laptop Webcam", "📱 Phone Camera", "ℹ About"])
    
    # ... (Keep Tab 1 and Tab 2 code exactly as they were) ...
    
    # ==================== LAPTOP WEBCAM TAB ====================
    with tab3:
        st.header("💻 Live Depth from Laptop Webcam")
        st.markdown("Use your laptop's built-in webcam for real-time depth estimation.")
        
        # Camera selection
        camera_index = st.selectbox(
            "Select Camera",
            [0, 1, 2],
            format_func=lambda x: f"Camera {x}" + (" (Default)" if x == 0 else ""),
            index=0
        )
        
        run_webcam = st.toggle("Start Webcam", key="webcam_toggle")
        
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
            
        if run_webcam:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for faster Windows camera access
            
            if not cap.isOpened():
                st.error(f"Could not open camera {camera_index}. Try a different camera index.")
            else:
                import time
                frame_count = 0
                start_time = time.time()
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame. Retrying...")
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
    
    # ==================== PHONE CAMERA TAB ====================
    with tab4:
        st.header("📱 Live Depth from Phone Camera")
        st.markdown("""
        1. Install *IP Webcam* (Android) or similar on your phone.
        2. Ensure phone and laptop are on the *same Wi-Fi*.
        3. Start the server on the phone and enter the IP address below (e.g., http://192.168.1.5:8080/video).
        """)
        
        # Input for the camera URL
        # Typical Android IP Webcam URL is http://<IP>:8080/video
        cam_url = st.text_input("Camera URL", "http://192.168.1.XX:8080/video")
        
        run_live = st.toggle("Start Live Stream", key="phone_toggle")
        
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
            
        if run_live:
            cap = cv2.VideoCapture(cam_url)
            
            if not cap.isOpened():
                st.error("Could not connect to phone camera. Check URL and Wi-Fi.")
            else:
                import time
                frame_count = 0
                start_time = time.time()
                
                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("Frame drop or stream ended.")
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

    # ==================== ABOUT TAB ====================
    with tab5:
        st.header("About This Application")
        st.markdown("""
        ### Model Architecture
        This depth estimation model uses a **ResNet-152** backbone as an encoder with a custom decoder featuring skip connections.
        
        **Key Features:**
        - **Encoder**: ResNet-152 for feature extraction
        - **Decoder**: Custom upsampling blocks with skip connections
        - **Output**: Depth map scaled to 0-10 meters
        
        ### Live Camera Options
        - **Laptop Webcam**: Use your built-in camera (Camera 0 is usually default)
        - **Phone Camera**: Use IP Webcam app to stream from your phone
        
        ### Performance Tips
        - Use **360p** resolution for fastest processing
        - GPU acceleration is enabled when CUDA is available
        - Lower resolution = Higher FPS
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and PyTorch")


if __name__ == "__main__":
    main()