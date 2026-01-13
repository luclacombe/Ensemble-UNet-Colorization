import cv2
import os
import numpy as np

# -------------------------------
# PATH CONFIGURATION
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

l_original_folder = os.path.join(BASE_DIR, "data", "BW")
la_folder = os.path.join(BASE_DIR, "Gen", "LA")
lb_folder = os.path.join(BASE_DIR, "Gen", "LB")
raw_output_folder = os.path.join(BASE_DIR, "Gen", "LA+LB")

# Ensure output folders exist
os.makedirs(raw_output_folder, exist_ok=True)

# Function to normalize an image
def normalize(image):
    image = image.astype(np.float32)
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_image.astype(np.uint8)

# Get list of image filenames
image_filenames = sorted([f for f in os.listdir(la_folder) if f.endswith('.png')])

# Process each image
for filename in image_filenames:
    la_path = os.path.join(la_folder, filename)
    lb_path = os.path.join(lb_folder, filename)
    l_path = os.path.join(l_original_folder, filename)
    raw_output_path = os.path.join(raw_output_folder, filename)
    # norm_output_path = os.path.join(norm_output_folder, filename)
    # test_path = os.path.join(test, filename)

    # Check if all required images exist
    if not (os.path.exists(la_path) and os.path.exists(lb_path) and os.path.exists(l_path)):
        print(f"Skipping {filename}: One or more source images are missing.")
        continue

    # Load images
    la_image = cv2.imread(la_path, cv2.IMREAD_UNCHANGED)
    lb_image = cv2.imread(lb_path, cv2.IMREAD_UNCHANGED)
    l_image = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)  # Load L as grayscale


    # Convert LA and LB images to LAB
    la_lab = cv2.cvtColor(la_image, cv2.COLOR_RGB2LAB)
    lb_lab = cv2.cvtColor(lb_image, cv2.COLOR_RGB2LAB)

    # Extract A and B channels
    a_channel = la_lab[:, :, 1]  # Extract A from LA
    b_channel = lb_lab[:, :, 2]  # Extract B from LB

    # Merge channels into LAB image
    lab_merged_raw = cv2.merge([l_image, a_channel, b_channel])

    # Convert LAB to RGB and save raw image
    final_image_raw = cv2.cvtColor(lab_merged_raw, cv2.COLOR_LAB2RGB)
    cv2.imwrite(raw_output_path, final_image_raw)

    ###############
    #### NORM #####
    ###############

    # --------- NORM 1 ----------#

    # normalized_image = normalize(lab_merged_raw)
    # final_image_norm = cv2.cvtColor(normalized_image, cv2.COLOR_LAB2RGB)
    # cv2.imwrite(norm_output_path, final_image_norm)

    ###

    # test = cv2.merge([0, a_channel, b_channel])
    # test_rgb = cv2.cvtColor(test, cv2.COLOR_LAB2RGB)
    # cv2.imwrite(test_path, test_rgb)

    print(f"Processed: {filename}")

print(f"Processed {len(image_filenames)} images. LAB images saved to: {raw_output_folder} and normalized images saved to: ")
