import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import matplotlib.pyplot as plt
import subprocess
import sys
import gc
from typing import Optional, Tuple

# ----------------------------
# Install missing dependencies
# ----------------------------
def install_package(package):
    """Install missing package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except Exception as e:
        st.error(f"Failed to install {package}: {str(e)}")
        return False

# Check and install required packages
required_packages = ['timm', 'opencv-python', 'pillow', 'matplotlib']
for package in required_packages:
    try:
        if package == 'timm':
            import timm
        elif package == 'opencv-python':
            import cv2
        elif package == 'pillow':
            from PIL import Image
        elif package == 'matplotlib':
            import matplotlib.pyplot as plt
    except ImportError:
        st.warning(f"📦 Installing missing dependency: {package}")
        if install_package(package):
            st.success(f"✅ Successfully installed {package}")
        else:
            st.error(f"❌ Failed to install {package}")

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(
    page_title="3D Image Generator Pro",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Simple Depth Estimation (Fallback)
# ----------------------------
def simple_depth_estimation(image: np.ndarray) -> np.ndarray:
    """
    Simple depth estimation using edge detection and blur
    as fallback when MiDaS is not available
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Edge detection (edges are typically closer)
    edges = cv2.Canny(gray, 50, 150)
    
    # Distance transform (closer pixels have higher values)
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    # Normalize
    if dist_transform.max() > 0:
        depth_map = dist_transform / dist_transform.max()
    else:
        # Fallback: center-focused depth
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        depth_map = 1 - np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
    
    return depth_map

# ----------------------------
# Cache configuration
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_midas_model(model_type: str = "MiDaS_small"):
    """
    Load MiDaS model with enhanced error handling and fallbacks
    """
    try:
        st.sidebar.info("🔄 Loading AI model... First time may take a minute.")
        
        # Try to install timm if not available
        try:
            import timm
        except ImportError:
            st.warning("Installing required 'timm' package...")
            install_package('timm')
            import timm
        
        # Set environment variable to avoid SSL issues
        os.environ['TORCH_HUB'] = '1'
        
        # Load model with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                midas = torch.hub.load(
                    "intel-isl/MiDaS", 
                    model_type,
                    trust_repo=True,
                    verbose=False,
                    skip_validation=True
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"❌ Model loading failed after {max_retries} attempts: {str(e)}")
                    return None, None, None
                else:
                    st.warning(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
        
        # Load transforms with fallback
        try:
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type == "DPT_Hybrid" or model_type == "DPT_Large":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform
        except:
            # Manual transform as fallback
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            transform = Compose([
                Resize(384),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device).eval()
        
        st.sidebar.success(f"✅ Model loaded on {device}")
        return midas, transform, device
        
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {str(e)}")
        st.info("🔧 Using fallback depth estimation method")
        return "fallback", "fallback", "fallback"

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Preprocess image for depth estimation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image

# ----------------------------
# Depth map generation
# ----------------------------
def generate_depth_map(
    image: Image.Image, 
    midas, 
    transform, 
    device
) -> Optional[np.ndarray]:
    """Generate depth map with fallback to simple method"""
    
    # Use fallback method if MiDaS failed to load
    if midas == "fallback":
        st.info("🔄 Using fallback depth estimation...")
        img_np = np.array(image)
        return simple_depth_estimation(img_np)
    
    if midas is None:
        return None
        
    try:
        img_np = np.array(image)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🔄 Preparing image...")
        progress_bar.progress(20)
        
        # Apply transform
        try:
            input_batch = transform(img_np)
            if isinstance(input_batch, dict):
                img_input = input_batch["image"]
            else:
                img_input = input_batch
        except:
            input_batch = transform({"image": img_np})
            img_input = input_batch["image"]
        
        img_input = img_input.to(device)
        if img_input.ndim == 3:
            img_input = img_input.unsqueeze(0)
        
        status_text.text("🔄 Running depth estimation...")
        progress_bar.progress(60)
        
        # Model inference
        with torch.no_grad():
            prediction = midas(img_input)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            elif isinstance(prediction, dict):
                prediction = prediction.get("out", prediction)
        
        status_text.text("🔄 Processing depth map...")
        progress_bar.progress(80)
        
        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_np = depth.cpu().numpy()
        
        progress_bar.progress(100)
        status_text.text("✅ Depth map generated!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return depth_np
        
    except Exception as e:
        st.error(f"❌ AI depth estimation failed: {str(e)}")
        st.info("🔄 Falling back to simple depth estimation...")
        img_np = np.array(image)
        return simple_depth_estimation(img_np)

# ----------------------------
# Enhanced 3D effect generation
# ----------------------------
def create_enhanced_3d_effect(
    image: np.ndarray, 
    depth_map: np.ndarray,
    effect_type: str = "parallax",
    intensity: float = 2.0,
    duration: float = 2.0
) -> Optional[str]:
    """Create 3D effects with configurable parameters"""
    try:
        h, w = image.shape[:2]
        
        # Normalize depth map
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min < 1e-8:
            depth_norm = np.ones_like(depth_map) * 0.5
        else:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        # Video settings
        fps = 20
        total_frames = int(fps * duration)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(temp_file.name, fourcc, fps, (w, h))
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(total_frames):
            progress = (i + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Generating frame {i+1}/{total_frames}")
            
            if effect_type == "parallax":
                t = i / total_frames
                shift = intensity * 8 * np.sin(2 * np.pi * t)
                
                x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
                x_map = (x_coords + shift * (1 - depth_norm)).astype(np.float32)
                y_map = y_coords.astype(np.float32)
                
            elif effect_type == "zoom":
                t = i / total_frames
                zoom_center_x, zoom_center_y = w // 2, h // 2
                
                x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
                dist_x = x_coords - zoom_center_x
                dist_y = y_coords - zoom_center_y
                
                zoom_factor = 1.0 + intensity * 0.05 * np.sin(2 * np.pi * t) * depth_norm
                new_x = zoom_center_x + dist_x / zoom_factor
                new_y = zoom_center_y + dist_y / zoom_factor
                
                x_map = new_x.astype(np.float32)
                y_map = new_y.astype(np.float32)
            
            # Apply transformation
            displacement = cv2.remap(
                image, 
                x_map, 
                y_map, 
                cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_REFLECT_101
            )
            output.write(displacement)
        
        output.release()
        progress_bar.empty()
        status_text.empty()
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"❌ 3D effect generation failed: {str(e)}")
        return None

# ----------------------------
# Visualization utilities
# ----------------------------
def create_depth_visualization(depth_map: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Create colored depth map visualization"""
    try:
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min < 1e-8:
            depth_normalized = np.ones_like(depth_map) * 0.5
        else:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        cmap = plt.cm.viridis
        depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        return depth_colored
    except:
        return np.zeros((100, 100, 3), dtype=np.uint8)

# ----------------------------
# Main UI
# ----------------------------
def main():
    st.title("🎮 3D Image Generator Pro")
    st.markdown("Transform your 2D images into stunning **3D spatial experiences**!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_type = st.selectbox(
            "Depth Model",
            ["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
            index=0,
            help="MiDaS_small: Fastest, DPT_Hybrid: Balanced, DPT_Large: Best quality"
        )
        
        effect_type = st.selectbox(
            "3D Effect",
            ["parallax", "zoom"],
            index=0
        )
        
        intensity = st.slider("Effect Intensity", 0.5, 3.0, 1.5, 0.1)
        duration = st.slider("Duration (seconds)", 1.0, 4.0, 2.0, 0.5)
        
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        **Best for:**
        - Portraits with clear faces
        - Landscapes with depth
        - Architecture
        
        **First run:** Installs dependencies automatically
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file:
            try:
                original_image = Image.open(uploaded_file)
                original_image = preprocess_image(original_image)
                st.image(original_image, caption="📷 Original Image", use_container_width=True)
                file_size = uploaded_file.size / 1024
                st.caption(f"📐 Size: {original_image.size[0]}×{original_image.size[1]} | 📊 File: {file_size:.1f} KB")
            except Exception as e:
                st.error(f"❌ Failed to load image: {str(e)}")
                uploaded_file = None
    
    with col2:
        if uploaded_file:
            st.subheader("🎯 Processing")
            
            with st.spinner("🔄 Loading AI model..."):
                midas, transform, device = load_midas_model(model_type)
            
            if midas is not None or midas == "fallback":
                depth_map = generate_depth_map(original_image, midas, transform, device)
                
                if depth_map is not None:
                    depth_colored = create_depth_visualization(depth_map, "viridis")
                    st.image(depth_colored, caption="🗺️ Depth Map", use_container_width=True)
                    
                    st.subheader("🎬 3D Animation")
                    with st.spinner("🔄 Creating 3D effect..."):
                        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                        video_path = create_enhanced_3d_effect(
                            img_cv, depth_map, effect_type, intensity, duration
                        )
                    
                    if video_path and os.path.exists(video_path):
                        st.success("✅ 3D animation created!")
                        st.video(video_path)
                        
                        st.subheader("📥 Download")
                        with open(video_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download 3D Animation (MP4)",
                                data=file,
                                file_name="3d_animation.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        
                        try:
                            os.unlink(video_path)
                        except:
                            pass

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    main()