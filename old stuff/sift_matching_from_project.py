import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths to the anchor and target image folders
anchor_root = 'data/anchor_no_bg'
target_root = 'data/camera_color_image_raw'

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Initialize the Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Helper function to isolate the fire hydrant in the anchor image
def isolate_hydrant(anchor_image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2HSV)

    # Define color range for fire hydrant (this range might need tuning)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    # Create a mask based on the color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hydrant)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        isolated_hydrant = anchor_image[y:y+h, x:x+w]
        return isolated_hydrant
    else:
        return anchor_image  # Return the original if no contours found

# Helper function to resize while maintaining aspect ratio
def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if aspect_ratio > 1:  # Width is greater than height
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:  # Height is greater than or equal to width
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
    else:
        return image  # Return original if it fits

    return cv2.resize(image, (new_width, new_height))

# Helper function to find potential fire hydrants in the target image
def find_potential_hydrants(target_image):
    hsv_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

    # Create a mask for potential fire hydrants
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Find contours for potential hydrants
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract rectangles around detected contours
    rectangles = [cv2.boundingRect(c) for c in contours]
    return rectangles

# Helper function to process and match images
def process_and_match_images(anchor_img, target_img):
    anchor_gray = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    keypoints_anchor, descriptors_anchor = sift.detectAndCompute(anchor_gray, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_gray, None)

    # Check if either of the descriptors are None
    if descriptors_anchor is None or descriptors_target is None:
        return [], None  # Return empty matches if descriptors are not found

    matches = bf.match(descriptors_anchor, descriptors_target)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, keypoints_target

# Process all anchor images with all target images
best_score = float('inf')
best_match_name = ""
best_matched_image = None
best_match_rectangle = None

for anchor_img_name in os.listdir(anchor_root):
    anchor_img_path = os.path.join(anchor_root, anchor_img_name)

    if os.path.isfile(anchor_img_path):
        anchor_image = cv2.imread(anchor_img_path)
        isolated_hydrant = isolate_hydrant(anchor_image)
        isolated_hydrant_resized = resize_image(isolated_hydrant, 640, 400)

        # Loop through target images directly in target_root
        for target_img_name in os.listdir(target_root):
            target_img_path = os.path.join(target_root, target_img_name)
            target_image = cv2.imread(target_img_path)

            # Find potential hydrants in the target image
            potential_hydrants = find_potential_hydrants(target_image)
            best_similarity_score = float('inf')

            # Check each potential fire hydrant
            for (x, y, w, h) in potential_hydrants:
                # Extract the region of interest
                roi = target_image[y:y+h, x:x+w]
                roi_resized = resize_image(roi, 640, 400)

                # Process and match the isolated hydrant with the ROI
                matches, keypoints_target = process_and_match_images(isolated_hydrant_resized, roi_resized)

                if matches:
                    top_n = 20
                    if len(matches) > top_n:
                        matches = matches[:top_n]
                    similarity_score = np.mean([match.distance for match in matches])

                    # Check for the best match
                    if similarity_score < best_similarity_score:
                        best_similarity_score = similarity_score
                        best_match_rectangle = (x, y, w, h)

            # If a best match is found, draw a rectangle around it
            if best_match_rectangle:
                x, y, w, h = best_match_rectangle
                cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the updated target image
            plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected Fire Hydrant: {anchor_img_name} in {target_img_name}")
            plt.axis('off')
            plt.show()

            # Track the overall best match across all target images
            if best_similarity_score < best_score:
                best_score = best_similarity_score
                best_match_name = target_img_name
                best_matched_image = target_image

# Highlight the best match
if best_matched_image is not None and best_matched_image.size > 0:
    print(f"Best match found: {best_match_name} with similarity score: {best_score}")
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(best_matched_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Best Match: {best_match_name} (Score: {best_score:.2f})")
    plt.axis('off')
    plt.show()
else:
    print("No matches found.")