# SLAM - Monocular Depth Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning application for real-time monocular depth estimation using a ResNet-152 encoder-decoder architecture with skip connections.

![Demo](docs/demo.gif)

## 🌟 Features

- **Image Depth Estimation**: Upload images and get instant depth maps
- **Video Processing**: Process video files with frame-by-frame depth estimation
- **Real-time Webcam**: Live depth estimation from laptop webcam
- **Phone Camera Support**: Stream from IP Webcam app for mobile depth estimation
- **Multiple Colormaps**: Choose from various visualization options (magma, plasma, viridis, etc.)
- **GPU Acceleration**: Automatic CUDA detection for faster inference
- **Adjustable Quality**: Multiple resolution presets for performance tuning

## 🏗️ Architecture

The model uses a **ResNet-152** backbone as an encoder with a custom decoder featuring skip connections (similar to U-Net architecture).

```
Input Image (RGB)
       │
       ▼
┌─────────────────┐
│   ResNet-152    │  ◄── Encoder (pretrained backbone)
│   Encoder       │
│                 │
│  Layer 0: 64ch  │──────────────────────────────────┐
│  Layer 1: 256ch │─────────────────────────┐        │
│  Layer 2: 512ch │────────────────┐        │        │
│  Layer 3: 1024ch│───────┐        │        │        │
│  Layer 4: 2048ch│       │        │        │        │
└────────┬────────┘       │        │        │        │
         │                │        │        │        │
         ▼                ▼        ▼        ▼        ▼
┌─────────────────────────────────────────────────────┐
│                    Decoder                          │
│                                                     │
│  UpBlock1: 2048+1024 → 1024  (skip from Layer 3)   │
│  UpBlock2: 1024+512  → 512   (skip from Layer 2)   │
│  UpBlock3: 512+256   → 256   (skip from Layer 1)   │
│  UpBlock4: 256+64    → 128   (skip from Layer 0)   │
│                                                     │
│  Final Conv: 128 → 1 (depth channel)               │
└────────┬────────────────────────────────────────────┘
         │
         ▼
   Depth Map (0-10m)
```

## 📊 Model Performance

Zero-shot evaluation on the iBims-1 benchmark (100 images):

| Metric | Value | Description |
|--------|-------|-------------|
| **RMSE** | 0.5848 m | Root Mean Square Error (lower is better) |
| **AbsRel** | 0.1365 | Absolute Relative Error (lower is better) |
| **δ < 1.25** | 81.56% | Accuracy threshold (higher is better) |
| **δ < 1.25²** | 95.64% | Accuracy threshold (higher is better) |
| **δ < 1.25³** | 98.63% | Accuracy threshold (higher is better) |

## 📁 Project Structure

```
SLAM/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
│
├── src/                  # Source code modules
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration settings
│   ├── model.py          # Neural network definitions
│   ├── inference.py      # Depth prediction logic
│   ├── video.py          # Video processing utilities
│   └── utils.py          # Helper functions
│
├── models/               # Model weights (not tracked in git)
│   └── .gitkeep
│
├── docs/                 # Documentation
│   └── demo.gif
│
└── tests/                # Unit tests
    └── __init__.py
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MridulBansal-123/SLAM.git
   cd SLAM
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model weights**
   
   Place `resnet152_depth_model.pth` in the parent directory or update the path in `src/config.py`.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 📖 Usage

### Image Upload
1. Go to the "📤 Upload Image" tab
2. Upload a JPG or PNG image
3. View the depth map visualization

### Video Processing
1. Go to the "🎬 Upload Video" tab
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Choose output mode (Depth Only or Side-by-Side)
4. Click "Process Video"
5. Download the processed video

### Laptop Webcam
1. Go to the "💻 Laptop Webcam" tab
2. Select your camera
3. Click "Start Webcam"
4. View real-time depth estimation

### Phone Camera
1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) on your Android phone
2. Connect phone and laptop to the same Wi-Fi
3. Start the IP Webcam server
4. Enter the URL (e.g., `http://192.168.1.100:8080/video`)
5. Click "Start Phone Camera"

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model settings
INPUT_SIZE = (640, 480)      # Input resolution
DEPTH_SCALE = 10.0           # Maximum depth in meters

# Default colormap
DEFAULT_COLORMAP = "magma"

# Resolution presets
RESOLUTION_PRESETS = {
    "360p (640x360)": (640, 360),
    "480p (854x480)": (854, 480),
    "720p (1280x720)": (1280, 720),
    "Original": None
}
```

## 🔧 API Usage

You can also use the depth estimator programmatically:

```python
from src.inference import DepthEstimator, colorize_depth
from PIL import Image

# Initialize estimator
estimator = DepthEstimator(model_path="path/to/model.pth")
estimator.load_model()

# Load and process image
image = Image.open("image.jpg").convert("RGB")
depth_map = estimator.estimate_depth(image)

# Visualize
depth_colored = colorize_depth(depth_map, colormap="magma")
```

## 📊 Performance

| Resolution | Device | FPS (approx) |
|------------|--------|--------------|
| 360p       | GPU    | 25-30        |
| 360p       | CPU    | 3-5          |
| 480p       | GPU    | 18-22        |
| 480p       | CPU    | 2-3          |
| 720p       | GPU    | 10-15        |
| 720p       | CPU    | 1-2          |

*Tested on NVIDIA RTX 3060 and Intel i7-10700*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- ResNet architecture by [He et al.](https://arxiv.org/abs/1512.03385)

## 📧 Contact

Dax Modi - [@daxmodi1](https://github.com/daxmodi1)
Mridul Bansal - [@MridulBansal-123](https://github.com/MridulBansal-123)

Project Link: [https://github.com/MridulBansal-123/SLAM](https://github.com/MridulBansal-123/SLAM)
