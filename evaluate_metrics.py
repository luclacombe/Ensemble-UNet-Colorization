import os
import cv2
import numpy as np
import pandas as pd
import lpips
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
loss_fn = lpips.LPIPS(net='vgg').eval().to(device)


def load_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def compute_lpips(rgb1, rgb2):
    rgb1_tensor = torch.tensor(rgb1, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    rgb2_tensor = torch.tensor(rgb2, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        return loss_fn(rgb1_tensor, rgb2_tensor).item()


def compute_emd(channel1, channel2):
    hist1 = np.histogram(channel1.flatten(), bins=256, range=(0, 256))[0]
    hist2 = np.histogram(channel2.flatten(), bins=256, range=(0, 256))[0]
    return wasserstein_distance(hist1 / hist1.sum(), hist2 / hist2.sum())


# -------------------------------
# PATH CONFIGURATION
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

paths = {
    "GT": os.path.join(BASE_DIR, "data", "Col"),
    "LA": os.path.join(BASE_DIR, "Gen", "LA"),
    "LB": os.path.join(BASE_DIR, "Gen", "LB"),
    "LAB": os.path.join(BASE_DIR, "Gen", "LAB"),
    "LA+LB": os.path.join(BASE_DIR, "Gen", "LA+LB")
}

# Collect valid images
valid_images = set()
for model in ["LA", "LB", "LAB", "LA+LB"]:
    valid_images.update(f for f in os.listdir(paths[model]) if f.lower().endswith('.png'))
image_files = [f for f in os.listdir(paths["GT"]) if f in valid_images]

metrics = []

for filename in tqdm(image_files):
    gt_rgb = load_image(os.path.join(paths["GT"], filename))
    if gt_rgb is None:
        continue

    generated_rgb = {}
    valid = True
    for model in ["LA", "LB", "LAB", "LA+LB"]:
        img = load_image(os.path.join(paths[model], filename))
        if img is None:
            valid = False
            break
        generated_rgb[model] = img

    if not valid:
        continue

    # Calculate metrics
    metric_entry = {"filename": filename}

    for model in generated_rgb:
        # MAE (average across RGB channels)
        metric_entry[f"{model}_MAE"] = np.abs(generated_rgb[model] - gt_rgb).mean()

        # LPIPS
        metric_entry[f"{model}_LPIPS"] = compute_lpips(generated_rgb[model], gt_rgb)

        # EMD (average across RGB channels)
        emd_values = []
        for c in range(3):  # For R, G, B channels
            emd_values.append(compute_emd(generated_rgb[model][..., c], gt_rgb[..., c]))
        metric_entry[f"{model}_EMD"] = np.mean(emd_values)

    metrics.append(metric_entry)

# Create and save report
df = pd.DataFrame(metrics)
csv_path = os.path.join(BASE_DIR, "Metrics", "colorization_evaluation_report.csv")
df.to_csv(csv_path, index=False)

print("Evaluation complete. Report saved.")
