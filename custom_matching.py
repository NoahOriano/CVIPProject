import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Paths to the anchor and target image folders
ANCHOR_ROOT = '/content/drive/MyDrive/R7020E-Project-Files/anchor_no_bg'
TARGET_ROOT = '/content/drive/MyDrive/R7020E-Project-Files/raw/test/camera_color_image_raw'

# Custom function to calculate Euclidean distance
def calculate_euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

# Custom feature matching function with ratio test
def custom_feature_matching(anchor_descriptors, target_descriptors, ratio_threshold=0.75):
    matches = []

    # For each descriptor in the anchor image, find the closest descriptor in the target image
    for i, anchor_desc in enumerate(anchor_descriptors):
        distances = []
        for j, target_desc in enumerate(target_descriptors):
            distance = calculate_euclidean_distance(anchor_desc, target_desc)
            distances.append((distance, j))  # Store distance and index of target descriptor

        # Sort distances and apply the ratio test
        distances = sorted(distances, key=lambda x: x[0])
        if len(distances) > 1:
            best_distance, best_index = distances[0]
            second_best_distance, _ = distances[1]
            
            # Apply ratio test to ensure distinctiveness of match
            if best_distance < ratio_threshold * second_best_distance:
                matches.append((i, best_index))  # (anchor descriptor index, target descriptor index)
        elif len(distances) == 1:
            # If there's only one match, accept it directly (no ratio test needed)
            matches.append((i, distances[0][1]))

    return matches

def display_matches(img1, kp1, img2, kp2, matches):
    """
    Display matched keypoints between two images using matplotlib.
    """
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, 
                                  [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=0) for m in matches],
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convert BGR to RGB for displaying in matplotlib
    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_img_rgb)
    plt.axis('off')
    plt.show()

def main():
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Load and process anchor images
    anchor_images = glob.glob(os.path.join(ANCHOR_ROOT, '*.png'))
    anchor_images.sort()
    anchor_keypoints = []
    anchor_descriptors = []

    # Detect SIFT features for anchor images
    for anchor_image_path in anchor_images:
        img = cv2.imread(anchor_image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Could not load image {anchor_image_path}")
            continue

        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            print(f"No descriptors found in anchor image {anchor_image_path}")
            continue

        anchor_keypoints.append(kp)
        anchor_descriptors.append(des)

    # Load and process target images
    target_images = glob.glob(os.path.join(TARGET_ROOT, '*.png'))
    target_images.sort()

    # Match features for each target image
    for target_image_path in target_images:
        img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Could not load image {target_image_path}")
            continue

        target_keypoints, target_descriptors = sift.detectAndCompute(img, None)
        if target_descriptors is None:
            print(f"No descriptors found in target image {target_image_path}")
            continue

        # Apply custom feature matching for each anchor image
        for anchor_idx, (kp_anchor, des_anchor) in enumerate(zip(anchor_keypoints, anchor_descriptors)):
            matches = custom_feature_matching(des_anchor, target_descriptors)

            # Display matched keypoints between the anchor and target images
            print(f"Displaying match between anchor image {anchor_idx} and target image {target_images.index(target_image_path)}")
            display_matches(cv2.imread(anchor_images[anchor_idx]), kp_anchor, 
                            cv2.imread(target_image_path), target_keypoints, matches)

if __name__ == "__main__":
    main()
