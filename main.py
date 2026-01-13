import os
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

import tensorflow as tf

# Check if MPS (Apple GPU) is available
if tf.config.experimental.list_physical_devices('GPU'):
    print("Running on Apple Metal GPU (MPS)")
else:
    print("No compatible GPU found. Running on CPU.")

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# -------------------------------
# 1. Global Variables
# -------------------------------
MODE = "LAB"

# Get the directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
bw_folder = os.path.join(BASE_DIR, "data", "BW")
col_folder = os.path.join(BASE_DIR, "data", "Col")

gen_folder = os.path.join(BASE_DIR, "Gen")
metric_folder = os.path.join(BASE_DIR, "Metrics")
models_folder = os.path.join(BASE_DIR, "Models")

# Create folders if not present
Path(gen_folder).mkdir(exist_ok=True, parents=True)
Path(metric_folder).mkdir(exist_ok=True, parents=True)
Path(models_folder).mkdir(exist_ok=True, parents=True)

# Store final colorized outputs in e.g. Gen/LA, Gen/LB, or Gen/LAB
mode_gen_folder = os.path.join(gen_folder, MODE)
Path(mode_gen_folder).mkdir(exist_ok=True, parents=True)

IMG_HEIGHT, IMG_WIDTH = 256, 256

# Training hyperparameters
EPOCHS = 20
BATCH_SIZE = 64
TRAIN_SUBSET_SIZE = 20000   # Max number of images used for training
VAL_SUBSET_SIZE = 200      # Max number of images used for validation

# Test set constants
TEST_SIZE = 2000           # Last 200 images are reserved for test
NUM_TO_COLORIZE = 2000      # We'll colorize 50 out of these 200

# We'll name the model file like "LAB_500_20_32.h5"
model_name = f"{MODE}_{TRAIN_SUBSET_SIZE}_{EPOCHS}_{BATCH_SIZE}.h5"
model_path = os.path.join(models_folder, model_name)

# -------------------------------
# 2. GPU Configuration (Optional)
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) detected. Memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU. Training may be slow.")

if gpus:
    # Enable MPS optimized execution (Metal Performance Shaders)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)

# -------------------------------
# 3. Build Reference Distribution
# -------------------------------
def build_ab_distribution(train_paths, col_folder, target_size=(256,256), nbins=256):
    """
    Builds the distribution(s) for the relevant channel(s) depending on MODE:
      - LA => only 'a' channel
      - LB => only 'b' channel
      - LAB => both 'a' and 'b' channels
    """
    hist_a = np.zeros(nbins, dtype=np.float64)
    hist_b = np.zeros(nbins, dtype=np.float64)
    total_pixels_a = 0
    total_pixels_b = 0

    for bw_path in train_paths:
        filename = os.path.basename(bw_path)
        col_path = os.path.join(col_folder, filename)
        col_img = cv2.imread(col_path)
        if col_img is None:
            continue
        
        col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
        col_img = cv2.resize(col_img, target_size)
        lab = cv2.cvtColor(col_img, cv2.COLOR_RGB2LAB)
        
        # If LA => only accumulate a
        # If LB => only accumulate b
        # If LAB => accumulate both
        if MODE in ["LA", "LAB"]:
            a_channel = lab[:,:,1].ravel()
            hist_a += np.histogram(a_channel, bins=nbins, range=(0,256))[0]
            total_pixels_a += a_channel.size
        
        if MODE in ["LB", "LAB"]:
            b_channel = lab[:,:,2].ravel()
            hist_b += np.histogram(b_channel, bins=nbins, range=(0,256))[0]
            total_pixels_b += b_channel.size
    
    # Convert histograms to CDF(s)
    cdf_a, cdf_b = None, None
    if MODE in ["LA", "LAB"] and total_pixels_a > 0:
        cdf_a = np.cumsum(hist_a) / total_pixels_a
    if MODE in ["LB", "LAB"] and total_pixels_b > 0:
        cdf_b = np.cumsum(hist_b) / total_pixels_b

    # Save histogram plot
    plt.figure(figsize=(6,4))
    if MODE in ["LA", "LAB"] and total_pixels_a > 0:
        plt.plot(hist_a, color='red', label='Histogram A')
    if MODE in ["LB", "LAB"] and total_pixels_b > 0:
        plt.plot(hist_b, color='blue', label='Histogram B')
    plt.title(f"Training Hist - Mode: {MODE}")
    plt.xlabel("Intensity (0..255)")
    plt.ylabel("Frequency")
    plt.legend()
    hist_plot_path = os.path.join(metric_folder, f"hist_{MODE}.png")
    plt.savefig(hist_plot_path)
    plt.close()
    print(f"Saved histogram plot to {hist_plot_path}")

    return cdf_a, cdf_b

def match_channel_to_cdf(channel, ref_cdf, nbins=256):
    if ref_cdf is None:
        # If we have no cdf (e.g. LB mode but we only used a?), return channel unchanged
        return channel
    flat = channel.ravel()
    
    hist_src, _ = np.histogram(flat, bins=nbins, range=(0,256))
    cdf_src = np.cumsum(hist_src).astype(np.float64)
    cdf_src /= cdf_src[-1]
    
    lut = np.zeros(nbins, dtype=np.uint8)
    for i in range(nbins):
        src_val = cdf_src[i]
        j = np.searchsorted(ref_cdf, src_val, side='left')
        if j >= nbins:
            j = nbins - 1
        lut[i] = j
    
    matched = lut[flat]
    matched = matched.reshape(channel.shape)
    return matched

def match_ab_to_reference(ab_img, cdf_a, cdf_b, nbins=256, alpha=0.5):
    """
    For LA => ab_img has shape (H,W,1) for 'a' channel
    For LB => ab_img has shape (H,W,1) for 'b' channel
    For LAB => shape (H,W,2)
    """
    if MODE == "LA":
        # ab_img is just a
        a_channel = ab_img[:,:,0]
        matched_a = match_channel_to_cdf(a_channel, cdf_a, nbins)
        # partial blending
        blended_a = (alpha * matched_a + (1 - alpha) * a_channel).astype(np.uint8)
        # shape (H,W,1)
        return np.expand_dims(blended_a, axis=-1)
    elif MODE == "LB":
        # ab_img is just b
        b_channel = ab_img[:,:,0]
        matched_b = match_channel_to_cdf(b_channel, cdf_b, nbins)
        blended_b = (alpha * matched_b + (1 - alpha) * b_channel).astype(np.uint8)
        return np.expand_dims(blended_b, axis=-1)
    else:
        # LAB => ab_img has shape (H,W,2)
        a_channel = ab_img[:,:,0]
        b_channel = ab_img[:,:,1]
        matched_a = match_channel_to_cdf(a_channel, cdf_a, nbins)
        matched_b = match_channel_to_cdf(b_channel, cdf_b, nbins)
        blended_a = (alpha * matched_a + (1 - alpha) * a_channel).astype(np.uint8)
        blended_b = (alpha * matched_b + (1 - alpha) * b_channel).astype(np.uint8)
        return np.stack([blended_a, blended_b], axis=-1)

# -------------------------------
# 4. Data & Preprocessing
# -------------------------------
def load_and_preprocess_pair(bw_path, col_path, target_size=(256, 256)):
    """
    If MODE == "LA": produce shape (H,W,1) for 'a'
    If MODE == "LB": produce shape (H,W,1) for 'b'
    If MODE == "LAB": produce shape (H,W,2) for 'ab'
    """
    bw_img = cv2.imread(bw_path, cv2.IMREAD_GRAYSCALE)
    if bw_img is None:
        raise ValueError(f"Could not read BW image: {bw_path}")
    bw_img = cv2.resize(bw_img, target_size)
    l = bw_img.astype("float32") / 255.0
    l = np.expand_dims(l, axis=-1)
    
    col_img = cv2.imread(col_path)
    if col_img is None:
        raise ValueError(f"Could not read color image: {col_path}")
    col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
    col_img = cv2.resize(col_img, target_size)
    lab = cv2.cvtColor(col_img, cv2.COLOR_RGB2LAB)

    if MODE == "LA":
        # only the a channel => shape (H,W,1)
        a_channel = lab[:,:,1].astype("float32")
        a_channel = (a_channel - 128) / 128.0
        a_channel = np.expand_dims(a_channel, axis=-1)
        return l, a_channel
    elif MODE == "LB":
        # only the b channel => shape (H,W,1)
        b_channel = lab[:,:,2].astype("float32")
        b_channel = (b_channel - 128) / 128.0
        b_channel = np.expand_dims(b_channel, axis=-1)
        return l, b_channel
    else:
        # LAB => both a,b => shape (H,W,2)
        ab = lab[:,:,1:].astype("float32")
        ab = (ab - 128) / 128.0
        return l, ab

def get_image_paths(folder):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]

# -------------------------------
# 5. U-Net Model
# -------------------------------
def build_light_unet(input_shape=(256, 256, 1)):
    """
    If MODE is LA or LB => final channels = 1
    If MODE is LAB => final channels = 2
    """
    if MODE in ["LA", "LB"]:
        output_channels = 1
    else:
        output_channels = 2

    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u2 = UpSampling2D((2, 2))(c3)
    u2 = concatenate([u2, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    
    u3 = UpSampling2D((2, 2))(c4)
    u3 = concatenate([u3, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u3)
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = Conv2D(output_channels, (1, 1), activation='tanh')(c5)
    return Model(inputs=inputs, outputs=outputs)

# -------------------------------
# 6. Colorization
# -------------------------------
def colorize_image(model, bw_path, target_size=(256, 256)):
    """
    Returns (l, predicted_channels).
    If MODE=LA => predicted_channels is shape (H,W,1) for 'a'
    If MODE=LB => shape (H,W,1) for 'b'
    If MODE=LAB => shape (H,W,2) for 'ab'
    """
    bw_img = cv2.imread(bw_path, cv2.IMREAD_GRAYSCALE)
    if bw_img is None:
        raise ValueError(f"Could not read BW image: {bw_path}")
    bw_img = cv2.resize(bw_img, target_size)
    l = bw_img.astype("float32") / 255.0
    l = np.expand_dims(l, axis=-1)
    
    l_input = np.expand_dims(l, axis=0)
    pred = model.predict(l_input)[0]  # shape (H,W,1 or 2)
    pred = (pred * 128) + 128
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    
    return l, pred

def colorize_and_save_histmatch(model, bw_path, output_path, cdf_a, cdf_b, target_size=(256,256), alpha=0.3):
    """
    If MODE=LA => we produce 'a' only => fill 'b' with 128 if needed
    If MODE=LB => we produce 'b' only => fill 'a' with 128
    If MODE=LAB => we produce both 'a','b'
    """
    l, pred = colorize_image(model, bw_path, target_size)
    matched_pred = match_ab_to_reference(pred, cdf_a, cdf_b, alpha=alpha)

    # Reconstruct Lab => RGB
    l_channel = (l[:,:,0] * 255).astype(np.uint8)

    if MODE == "LA":
        # 'a' is matched_pred[:,:,0]
        a_channel = matched_pred[:,:,0]
        # fill b=128
        b_channel = np.full_like(a_channel, 128, dtype=np.uint8)
    elif MODE == "LB":
        # 'b' is matched_pred[:,:,0]
        b_channel = matched_pred[:,:,0]
        # fill a=128
        a_channel = np.full_like(b_channel, 128, dtype=np.uint8)
    else:  # LAB
        a_channel = matched_pred[:,:,0]
        b_channel = matched_pred[:,:,1]

    lab = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    lab[:,:,0] = l_channel
    lab[:,:,1] = a_channel
    lab[:,:,2] = b_channel
    
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# -------------------------------
# 7. Main Script
# -------------------------------
def main():
    # Step 1: Gather all grayscale image paths
    all_bw_paths = get_image_paths(bw_folder)
    total_paths = len(all_bw_paths)
    if total_paths < TEST_SIZE:
        raise ValueError(f"Not enough images ({total_paths}) to reserve {TEST_SIZE} for test.")
    
    # Reserve last 200 images for test
    test_bw_paths = all_bw_paths[-TEST_SIZE:]
    train_val_bw_paths = all_bw_paths[:-TEST_SIZE]
    print(f"[{MODE}] Found {total_paths} total images. Using last {TEST_SIZE} for test set.")
    print(f"[{MODE}] Training/Validation set size: {len(train_val_bw_paths)}")

    # Train/val split
    train_bw_paths, val_bw_paths = train_test_split(train_val_bw_paths, test_size=0.1, random_state=42)
    train_bw_paths = train_bw_paths[:TRAIN_SUBSET_SIZE]
    val_bw_paths   = val_bw_paths[:VAL_SUBSET_SIZE]
    print(f"[{MODE}] Train set: {len(train_bw_paths)}, Validation set: {len(val_bw_paths)}, Test set: {len(test_bw_paths)}")

    # Build dataset
    def data_generator(bw_list, batch_size):
        while True:
            np.random.shuffle(bw_list)
            for i in range(0, len(bw_list), batch_size):
                batch = bw_list[i:i+batch_size]
                l_batch, chan_batch = [], []
                for bw_path in batch:
                    filename = os.path.basename(bw_path)
                    col_path = os.path.join(col_folder, filename)
                    try:
                        l, c = load_and_preprocess_pair(bw_path, col_path)
                        l_batch.append(l)
                        chan_batch.append(c)
                    except Exception as e:
                        print(f"Skipping {bw_path}: {e}")
                if l_batch and chan_batch:
                    yield np.array(l_batch), np.array(chan_batch)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_bw_paths, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            # shape depends on MODE, but we can just set to unknown last dim
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, None), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_bw_paths, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, None), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(train_bw_paths) // BATCH_SIZE
    val_steps = len(val_bw_paths) // BATCH_SIZE

    # If model file exists, load. Otherwise, train new.
    if os.path.exists(model_path):
        print(f"[{MODE}] Found existing model '{model_name}' in '{models_folder}'. Loading it.")
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(), loss='mse')
    else:
        print(f"[{MODE}] No existing model found. Building and training new model: {model_name}")
        model = build_light_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
        model.compile(optimizer=Adam(), loss='mse')

        # Build distribution from train set
        print(f"[{MODE}] Building distribution for histogram matching...")
        cdf_a, cdf_b = build_ab_distribution(train_bw_paths, col_folder)

        # Train
        print(f"[{MODE}] Training for {EPOCHS} epochs, batch size {BATCH_SIZE}...")
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=EPOCHS
        )

        # Save model
        model.save(model_path)
        print(f"[{MODE}] Model saved to {model_path}")

        # Save loss curves
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], marker='o', label='Training Loss')
        plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
        plt.title(f"{MODE} Loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(metric_folder, f"loss_curve_{MODE}.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[{MODE}] Saved loss curve to {loss_curve_path}")

    # Regardless of training or loading, we need cdf_a, cdf_b
    # If we haven't built them yet, do it now.
    if not os.path.exists(os.path.join(metric_folder, f"hist_{MODE}.png")):
        print(f"[{MODE}] Building distribution from train set (no hist found).")
        cdf_a, cdf_b = build_ab_distribution(train_bw_paths, col_folder)
    else:
        print(f"[{MODE}] Recomputing distribution from train set for matching.")
        cdf_a, cdf_b = build_ab_distribution(train_bw_paths, col_folder)

    # Colorize from test set
    test_count = len(test_bw_paths)
    if test_count < NUM_TO_COLORIZE:
        chosen_test_files = test_bw_paths
    else:
        chosen_test_files = random.sample(test_bw_paths, NUM_TO_COLORIZE)

    print(f"[{MODE}] Colorizing {len(chosen_test_files)} test images => '{mode_gen_folder}'")
    for idx, bw_path in enumerate(chosen_test_files):
        filename = os.path.basename(bw_path)
        output_path = os.path.join(mode_gen_folder, filename)
        colorize_and_save_histmatch(model, bw_path, output_path, cdf_a, cdf_b, alpha=0.3)
        print(f"[{MODE}] Processed {idx+1}/{len(chosen_test_files)} => '{output_path}'")

    print(f"[{MODE}] Done.")

if __name__ == "__main__":
    main()
