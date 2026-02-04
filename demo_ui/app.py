# app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
import os
from nlp_reasoner.nlp_reasoner import generate_narration


st.set_page_config(page_title="Mission-Aware Scene Narrator", layout="wide")

st.title("🛰️ Mission-Aware Scene Narrator")
st.write("Semantic segmentation–driven terrain understanding")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Mission Settings")
mission = st.sidebar.selectbox(
    "Select Mission Objective",
    ["speed", "energy", "safety", "exploration"]
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mask_dir = os.path.join(ROOT_DIR, "predictions", "masks")
color_mask_dir = os.path.join(ROOT_DIR, "predictions", "masks_color")

if not os.path.exists(mask_dir):
    st.error(f"Mask directory not found at {mask_dir}. Please run inference first.")
    st.stop()

mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
selected_mask_file = st.sidebar.selectbox("Select Scene", mask_files)

# Extract base ID for the real image (e.g., 0000066 from 0000066_pred.png)
base_id = selected_mask_file.split("_")[0]
real_image_path = os.path.join(ROOT_DIR, "Offroad_Segmentation_testImages", "Color_Images", f"{base_id}.png")

if os.path.exists(real_image_path):
    st.sidebar.image(real_image_path, caption="Original Camera View", use_container_width=True)
else:
    st.sidebar.warning(f"Real image not found at {real_image_path}")

# -------------------------
# Load masks
# -------------------------
# Load raw mask for calculations
mask_path = os.path.join(mask_dir, selected_mask_file)
mask = cv2.imread(mask_path, 0)

# Load colored mask for display
# Mapping: 0000062_pred.png -> 0000062_pred_color.png
base_name = os.path.splitext(selected_mask_file)[0]
color_mask_path = os.path.join(color_mask_dir, f"{base_name}_color.png")

# Fallback in case naming differs: 
# test_segmentation.py uses os.path.join(masks_color_dir, f'{base_name}_pred_color.png')
# Wait, let me check the naming in predictions/masks_color
if not os.path.exists(color_mask_path):
    # Try another common pattern from the script
    color_mask_path = os.path.join(color_mask_dir, f"{base_name.replace('_pred','')}_pred_color.png")

if os.path.exists(color_mask_path):
    color_mask = cv2.imread(color_mask_path)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
else:
    color_mask = None
    st.warning(f"Colored mask not found at {color_mask_path}")

total_pixels = mask.size

# Class IDs (Mapping from test_segmentation.py)
CLASS_IDS = {
    "Rocks": 7,
    "Logs": 6,
    "Ground Clutter": 5,
    "Landscape": 8,
    "Sky": 9,
    "Vegetation": [1, 2, 3, 4] # Trees, Lush Bushes, Dry Grass, Dry Bushes
}

stats = {}
for name, cid in CLASS_IDS.items():
    if isinstance(cid, list):
        count = sum(np.sum(mask == c) for c in cid)
    else:
        count = np.sum(mask == cid)
    stats[name.lower()] = round((count / total_pixels) * 100, 2)

stats["obstacles"] = round(stats["rocks"] + stats["logs"], 2)

# -------------------------
# Generate narration
# -------------------------
# Map stats to what nlp_reasoner expects (it uses sand for ground clutter)
narration_stats = stats.copy()
narration_stats["sand"] = stats.get("ground clutter", 0)
narration = generate_narration(narration_stats, mission)

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Segmentation Visualization")
    if color_mask is not None:
        st.image(color_mask, use_container_width=True, caption=f"Scene: {selected_mask_file}")
    else:
        # If no color mask, at least normalize the grayscale one for visibility
        normalized_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        st.image(normalized_mask, use_container_width=True, caption="Normalized Mask (Raw IDs scaled)")

with col2:
    st.subheader("Scene Analysis")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Obstacle Density", f"{stats['obstacles']}%")
    metric_cols[1].metric("Drivable Area", f"{stats['landscape']}%")
    
    with st.expander("Detailed Composition"):
        for k, v in stats.items():
            if k not in ["obstacles"]:
                st.write(f"**{k.title()}**: {v}%")

    st.subheader("Mission Narration")
    st.info(narration)

    # Color Legend
    st.markdown("### 🎨 Legend")
    legend_md = """
| Color | Class |
| :--- | :--- |
| ![#808080](https://via.placeholder.com/15/808080/000000?text=+) Gray | Rocks |
| ![#8B4513](https://via.placeholder.com/15/8B4513/000000?text=+) Brown | Logs |
| ![#A0522D](https://via.placeholder.com/15/A0522D/000000?text=+) Sienna | Landscape |
| ![#87CEEB](https://via.placeholder.com/15/87CEEB/000000?text=+) Sky Blue | Sky |
| ![#228B22](https://via.placeholder.com/15/228B22/000000?text=+) Green | Vegetation |
"""
    st.markdown(legend_md)

st.markdown("---")
st.caption("TechnoMania 2.0 | NLP Elective Track")