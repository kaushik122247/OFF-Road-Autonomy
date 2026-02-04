import cv2
import os
import json
import csv
import numpy as np
from tqdm import tqdm

MASK_DIR = "predictions/masks"
OUTPUT_DIR = "mask_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_MAP = {
    0: "background",
    1: "trees",
    2: "lush_bushes",
    3: "dry_grass",
    4: "dry_bushes",
    5: "ground_clutter",
    6: "logs",
    7: "rocks",
    8: "landscape",
    9: "sky"
}

all_stats = []

if not os.path.exists(MASK_DIR):
    print(f"Directory {MASK_DIR} does not exist. Please run test_segmentation.py first.")
    exit(1)

files = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]
print(f"Processing {len(files)} masks...")

for fname in tqdm(files):
    path = os.path.join(MASK_DIR, fname)
    mask = cv2.imread(path, 0)
    
    if mask is None:
        continue

    total_pixels = mask.size
    stats = {"image": fname}

    # Percentage for each class
    for cid, cname in CLASS_MAP.items():
        pct = np.sum(mask == cid) / total_pixels * 100
        stats[f"{cname}_pct"] = round(pct, 2)

    # Obstacle density (Logs + Rocks)
    OBSTACLE_CLASSES = [6, 7]
    obstacle_pixels = sum(np.sum(mask == cid) for cid in OBSTACLE_CLASSES)
    stats["obstacle_density_pct"] = round(obstacle_pixels / total_pixels * 100, 2)

    # Count of rock components
    rock_binary = np.uint8(mask == 7)
    num_labels_rocks, _ = cv2.connectedComponents(rock_binary)
    stats["rocks_count"] = num_labels_rocks - 1

    # Largest drivable area (Landscape)
    landscape_binary = np.uint8(mask == 8)
    num_labels_land, labels_land = cv2.connectedComponents(landscape_binary)
    
    max_area = 0
    for lbl in range(1, num_labels_land):
        area = np.sum(labels_land == lbl)
        max_area = max(max_area, area)
    
    stats["largest_drivable_area_pct"] = round(max_area / total_pixels * 100, 2)
    
    all_stats.append(stats)

if all_stats:
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "scene_stats.json")
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "scene_stats.csv")
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"\nStats saved to {OUTPUT_DIR}/")
else:
    print("No masks processed.")
