# app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from nlp_reasoner.nlp_reasoner import generate_narration

# ============================================================================
# Model Definitions (Must match training)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ============================================================================
# Constants & Helpers
# ============================================================================

COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(COLOR_PALETTE)):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Backbone
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.eval()
    backbone_model.to(device)
    
    # Load Head
    # Target dimensions used in training
    w_train = int(((960 / 2) // 14) * 14)
    h_train = int(((540 / 2) // 14) * 14)
    
    classifier = SegmentationHeadConvNeXt(
        in_channels=384, # DINOv2 small embedding dim
        out_channels=10,
        tokenW=w_train // 14,
        tokenH=h_train // 14
    )
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(ROOT_DIR, "segmentation_head.pth")
    if os.path.exists(model_path):
        classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    
    return backbone_model, classifier, device

def run_inference(image_np, backbone, head, device):
    # Prepare image
    w_train = int(((960 / 2) // 14) * 14)
    h_train = int(((540 / 2) // 14) * 14)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h_train, w_train)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        logits = head(features)
        outputs = F.interpolate(logits, size=(image_np.shape[0], image_np.shape[1]), mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(outputs[0], dim=0).cpu().numpy().astype(np.uint8)
    
    return pred_mask

# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Mission-Aware Scene Narrator", layout="wide")

st.title("🛰️ Mission-Aware Scene Narrator")
st.write("Semantic segmentation–driven terrain understanding")

# Load models
with st.spinner("Loading AI models..."):
    backbone, head, device = load_models()

# Sidebar
st.sidebar.header("Mission Settings")
mission = st.sidebar.selectbox("Select Mission Objective", ["speed", "energy", "safety", "exploration"])

input_mode = st.sidebar.radio("Input Source", ["Dataset", "Camera"])

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mask_dir = os.path.join(ROOT_DIR, "predictions", "masks")
color_mask_dir = os.path.join(ROOT_DIR, "predictions", "masks_color")

selected_image_np = None
selected_mask_np = None
selected_color_mask_np = None
source_name = ""

if input_mode == "Dataset":
    if not os.path.exists(mask_dir):
        st.error("Predictions not found. Please run test_segmentation.py first.")
        st.stop()
    
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    selected_mask_file = st.sidebar.selectbox("Select Scene", mask_files)
    source_name = selected_mask_file

    # Load Real Image
    base_id = selected_mask_file.split("_")[0]
    real_path = os.path.join(ROOT_DIR, "Offroad_Segmentation_testImages", "Color_Images", f"{base_id}.png")
    if os.path.exists(real_path):
        img_bgr = cv2.imread(real_path)
        selected_image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.sidebar.image(selected_image_np, caption="Original Camera View", use_container_width=True)
    
    # Load Pre-calculated Masks
    mask_path = os.path.join(mask_dir, selected_mask_file)
    selected_mask_np = cv2.imread(mask_path, 0)
    
    color_path = os.path.join(color_mask_dir, f"{os.path.splitext(selected_mask_file)[0].replace('_pred','')}_pred_color.png")
    if not os.path.exists(color_path):
        color_path = os.path.join(color_mask_dir, f"{os.path.splitext(selected_mask_file)[0]}_color.png")
        
    if os.path.exists(color_path):
        col_bgr = cv2.imread(color_path)
        selected_color_mask_np = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)

else:
    cam_image = st.camera_input("Capture Terrain")
    if cam_image:
        img_pil = Image.open(cam_image)
        selected_image_np = np.array(img_pil)
        source_name = "Camera Stream"
        
        with st.spinner("Analyzing terrain..."):
            selected_mask_np = run_inference(selected_image_np, backbone, head, device)
            selected_color_mask_np = mask_to_color(selected_mask_np)

if selected_image_np is not None and selected_mask_np is not None:
    # Calculations
    total_pixels = selected_mask_np.size
    CLASS_IDS = {
        "Rocks": 7, "Logs": 6, "Ground Clutter": 5, "Landscape": 8, "Sky": 9,
        "Vegetation": [1, 2, 3, 4]
    }
    
    stats = {}
    for name, cid in CLASS_IDS.items():
        if isinstance(cid, list):
            count = sum(np.sum(selected_mask_np == c) for c in cid)
        else:
            count = np.sum(selected_mask_np == cid)
        stats[name.lower()] = round((count / total_pixels) * 100, 2)
    
    stats["obstacles"] = round(stats["rocks"] + stats["logs"], 2)
    narration_stats = stats.copy()
    narration_stats["sand"] = stats.get("ground clutter", 0)
    narration = generate_narration(narration_stats, mission)

    # UI Layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Segmentation Visualization")
        if selected_color_mask_np is not None:
            st.image(selected_color_mask_np, use_container_width=True, caption=f"Analyzed: {source_name}")
        else:
            norm = cv2.normalize(selected_mask_np, None, 0, 255, cv2.NORM_MINMAX)
            st.image(norm, use_container_width=True, caption="Raw Binary Output")

    with col2:
        st.subheader("Scene Analysis")
        mc = st.columns(2)
        mc[0].metric("Obstacle Density", f"{stats['obstacles']}%")
        mc[1].metric("Drivable Area", f"{stats['landscape']}%")
        
        with st.expander("Detailed Composition"):
            for k, v in stats.items():
                if k != "obstacles": st.write(f"**{k.title()}**: {v}%")
        
        st.subheader("Mission Narration")
        st.info(narration)
        
        st.markdown("### 🎨 Legend")
        st.markdown("""
| Color | Class |
| :--- | :--- |
| ![#808080](https://placehold.co/15x15/808080/808080?text=+) Gray | Rocks |
| ![#8B4513](https://placehold.co/15x15/8B4513/8B4513?text=+) Brown | Logs |
| ![#A0522D](https://placehold.co/15x15/A0522D/A0522D?text=+) Sienna | Landscape |
| ![#87CEEB](https://placehold.co/15x15/87CEEB/87CEEB?text=+) Sky | Sky |
| ![#228B22](https://placehold.co/15x15/228B22/228B22?text=+) Green | Vegetation |
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("TechnoMania 2.0 | OFF-Road Autonomy | NLP Reasoning Engine")