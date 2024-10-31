import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Paths to the anchor and target image folders
ANCHOR_ROOT = 'data\\anchor_no_bg'
TARGET_ROOT = 'output\\matching_output'

# Custom function to calculate Euclidean distance
def calculate_euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

# Custom feature matching function with ratio test
def custom_feature_matching(anchor_descriptors, target_descriptors, ratio_threshold=0.8):
    matches = []
    for i, anchor_desc in enumerate(anchor_descriptors):
        distances = []
        for j, target_desc in enumerate(target_descriptors):
            distance = calculate_euclidean_distance(anchor_desc, target_desc)
            distances.append((distance, j))
        distances = sorted(distances, key=lambda x: x[0])
        if len(distances) > 1:
            best_distance, best_index = distances[0]
            second_best_distance, _ = distances[1]
            if best_distance < ratio_threshold * second_best_distance:
                matches.append((i, best_index, best_distance))  # Store (anchor index, target index, score)
        elif len(distances) == 1:
            matches.append((i, distances[0][1], distances[0][0]))  # Only one match available

    return matches

# Function to display matches along with similarity scores
def display_matches(img1, kp1, img2, kp2, matches):
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, 
                                  [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=0) for m in matches],
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    
    # Display the matched image
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_img_rgb)
    plt.axis('off')
    plt.show()

    # Print similarity scores
    print("Similarity scores for matched keypoints:")
    for match in matches:
        print(f"Anchor Keypoint Index: {match[0]}, Target Keypoint Index: {match[1]}, Similarity Score: {match[2]:.2f}")

# Function to apply color segmentation for red objects (like the fire extinguisher)
def segment_red_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    result = cv2.bitwise_and(image, image, mask=mask)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

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
        img = cv2.imread(anchor_image_path)
        if img is None:
            print(f"Error: Could not load image {anchor_image_path}")
            continue
        img = segment_red_color(img)
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
        img = cv2.imread(target_image_path)
        if img is None:
            print(f"Error: Could not load image {target_image_path}")
            continue
        img = segment_red_color(img)
        target_keypoints, target_descriptors = sift.detectAndCompute(img, None)
        if target_descriptors is None:
            print(f"No descriptors found in target image {target_image_path}")
            continue

        # Apply custom feature matching for each anchor image
        for anchor_idx, (kp_anchor, des_anchor) in enumerate(zip(anchor_keypoints, anchor_descriptors)):
            matches = custom_feature_matching(des_anchor, target_descriptors)

            # Display matched keypoints between the anchor and target images
            print(f"\nDisplaying match between anchor image {anchor_idx} and target image {target_images.index(target_image_path)}")
            display_matches(cv2.imread(anchor_images[anchor_idx]), kp_anchor, 
                            cv2.imread(target_image_path), target_keypoints, matches)

if __name__ == "__main__":
    main()
