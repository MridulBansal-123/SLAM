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
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
# ==================== MODEL DEFINITION ====================

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


# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model():
    """Load the pre-trained depth estimation model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetDepthModel()
    
    # Load the trained weights
    model_path = "resnet152_depth_model.pth"
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        st.success(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device
    
    model = model.to(device)
    model.eval()
    return model, device


def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize to standard size
    image_resized = image.resize((640, 480))
    
    # Convert to Tensor & Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_resized).unsqueeze(0)
    return input_tensor, image.size


def predict_depth(model, device, input_tensor, original_size):
    """Run depth estimation"""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
        # Resize prediction back to original image size
        pred = F.interpolate(pred, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=True)
    
    depth_map = pred.squeeze().cpu().numpy()
    return depth_map


def create_depth_visualization(depth_map, colormap='magma'):
    """Create a colored depth map visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(depth_map, cmap=colormap)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Depth (meters)', fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return buf


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


# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Depth Estimation - ResNet152",
        page_icon="🔭",
        layout="wide"
    )
    
    st.title("🔭 Monocular Depth Estimation")
    st.markdown("""
    ### ResNet-152 Based Depth Estimation Model
    This application uses a pre-trained ResNet-152 encoder-decoder model to estimate depth from single RGB images.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    colormap = st.sidebar.selectbox(
        "Depth Colormap",
        ["magma", "plasma", "inferno", "viridis", "gray", "gray_r", "jet"],
        index=0
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'resnet152_depth_model.pth' exists.")
        return
    
    st.sidebar.info(f"🖥️ Device: {device}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Sample Images", "📤 Upload Image", "🎬 Upload Video", "ℹ️ About"])
    
    with tab1:
        st.header("Sample Images Depth Estimation")
        
        col1, col2 = st.columns(2)
        
        # Image 1
        with col1:
            st.subheader("Image 1")
            try:
                img1 = Image.open("1.jpg").convert('RGB')
                st.image(img1, caption="Original Image 1", use_container_width=True)
                
                if st.button("🔍 Estimate Depth - Image 1", key="btn1"):
                    with st.spinner("Processing..."):
                        input_tensor, original_size = preprocess_image(img1)
                        depth_map = predict_depth(model, device, input_tensor, original_size)
                        
                        st.subheader("Depth Map")
                        depth_viz = create_depth_visualization(depth_map, colormap)
                        st.image(depth_viz, caption="Predicted Depth Map", use_container_width=True)
                        
                        # Stats
                        st.metric("Min Depth", f"{depth_map.min():.2f} m")
                        st.metric("Max Depth", f"{depth_map.max():.2f} m")
                        st.metric("Mean Depth", f"{depth_map.mean():.2f} m")
            except FileNotFoundError:
                st.warning("Image '1.jpg' not found in the project directory.")
        
        # Image 2
        with col2:
            st.subheader("Image 2")
            try:
                img2 = Image.open("2.jpg").convert('RGB')
                st.image(img2, caption="Original Image 2", use_container_width=True)
                
                if st.button("🔍 Estimate Depth - Image 2", key="btn2"):
                    with st.spinner("Processing..."):
                        input_tensor, original_size = preprocess_image(img2)
                        depth_map = predict_depth(model, device, input_tensor, original_size)
                        
                        st.subheader("Depth Map")
                        depth_viz = create_depth_visualization(depth_map, colormap)
                        st.image(depth_viz, caption="Predicted Depth Map", use_container_width=True)
                        
                        # Stats
                        st.metric("Min Depth", f"{depth_map.min():.2f} m")
                        st.metric("Max Depth", f"{depth_map.max():.2f} m")
                        st.metric("Mean Depth", f"{depth_map.mean():.2f} m")
            except FileNotFoundError:
                st.warning("Image '2.jpg' not found in the project directory.")
    
    with tab2:
        st.header("Upload Your Own Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an RGB image for depth estimation"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info(f"Image Size: {image.size[0]} x {image.size[1]}")
            
            with col2:
                st.subheader("Depth Estimation")
                
                if st.button("🔍 Estimate Depth", key="btn_upload"):
                    with st.spinner("Processing..."):
                        input_tensor, original_size = preprocess_image(image)
                        depth_map = predict_depth(model, device, input_tensor, original_size)
                        
                        depth_viz = create_depth_visualization(depth_map, colormap)
                        st.image(depth_viz, caption="Predicted Depth Map", use_container_width=True)
                        
                        # Statistics
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Min Depth", f"{depth_map.min():.2f} m")
                        with col_stat2:
                            st.metric("Max Depth", f"{depth_map.max():.2f} m")
                        with col_stat3:
                            st.metric("Mean Depth", f"{depth_map.mean():.2f} m")
    
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
    
    with tab4:
        st.header("About This Application")
        st.markdown("""
        ### Model Architecture
        This depth estimation model uses a **ResNet-152** backbone as an encoder with a custom decoder featuring skip connections (similar to U-Net architecture).
        
        **Key Features:**
        - **Encoder**: Pre-trained ResNet-152 for feature extraction
        - **Decoder**: Custom upsampling blocks with skip connections
        - **Output**: Depth map scaled to 0-10 meters
        
        ### How It Works
        1. **Input**: RGB image (resized to 640x480 for processing)
        2. **Encoding**: Features extracted at multiple scales using ResNet-152
        3. **Decoding**: Features upsampled with skip connections for detail preservation
        4. **Output**: Dense depth map at original resolution
        
        ### Colormap Options
        - **Magma/Plasma/Inferno**: Perceptually uniform colormaps (recommended)
        - **Viridis**: Good for general visualization
        - **Gray/Gray_r**: Grayscale visualization (closer = brighter or darker)
        - **Jet**: Rainbow colormap (not recommended for accurate perception)
        
        ### References
        - NYU Depth V2 Dataset
        - ResNet-152 (He et al., 2016)
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and PyTorch")


if __name__ == "__main__":
    main()
