import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

import sift_matching_algorithm
import sift_algorithm

print("SIFT Matching Algorithm")
print("Begining initialization...")

show_debug_images = True
# Padding is used to create a larger region of interest (ROI) around the detected object to ensure the entire object is captured
# Modiffy these values based on the object in question.
image_roi_padding = [1, 0.3, 0.3, 0.3] # Top, right, bottom, left padding as a ratio of the ROI size
# Resize the anchor image, the smaller the value, the smaller the anchor image. 
# This is set small as the anchor resolution is high compared to the target images
anchor_scaling = 0.125


# Define paths to the anchor and target image folders
anchor_root = 'data/Anchor'
target_root = 'data/camera_color_image_raw'

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect the SIFT features for the anchor images
anchor_paths = glob.glob(anchor_root + '/*.png')
anchor_paths.sort()
anchor_keypoints = np.array([])

def median_blur(image, kernel_size):
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer.")
    
    # Define the padding size
    pad_size = kernel_size // 2
    
    # Pad the image to handle edges
    padded_image = np.pad(image, pad_size, mode='edge')
    
    # Prepare an output array of the same shape as the original image
    result = np.zeros_like(image)
    
    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the kernel region
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size]
            
            # Compute the median and set it to the result
            result[i, j] = np.median(kernel_region)
    
    return result

# Helper function to detect SIFT features for the target images
def detect_sift_features_from_path(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def detect_sift_features(image):
    # If the image is not grayscale, convert it to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to float32 for SIFT
    u_image = np.float32(image)
    kp, des = sift_algorithm.computeKeypointsAndDescriptors(u_image)
    return kp, des

def match_sift_features(anchor_descriptors, target_descriptors):
    # Use sift_matching_algorithm to match the descriptors
    matches = sift_matching_algorithm.custom_feature_matching(anchor_descriptors, target_descriptors)
    # Apply ratio test
    good = []
    good_without_list = []
    # Ensure there are at least 2 matches
    if len(matches) < 2:
        return good_without_list, good
    if(len(matches[0]) < 2):
        return good_without_list, good
    # Handling of good matches is done by the sift_matching_algorithm (using the ratio test)
    for match in matches:
        good.append([match])
        good_without_list.append(match)
    return good_without_list, good

# Helper function to get the mask for the anchor images
def get_anchor_mask(image):
    # Check if image has an alpha channel (transparency)
    if image.shape[2] == 4:
        # Extract the alpha channel to create a mask
        alpha_channel = image[:, :, 3]
        mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
        # Convert to grayscale for SIFT if the image has RGB channels
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        # Assume a single-color background and use color-based thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create mask by thresholding
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Apply the mask to focus on the object
    foreground = cv2.bitwise_and(gray, gray, mask=mask)
    return foreground, mask

# Initiaize a grouped image to show all the anchor images together
img = cv2.imread(anchor_paths[0], cv2.IMREAD_COLOR)
h, w = img.shape[:2]
h = int(h * anchor_scaling)
w = int(w * anchor_scaling)
img = cv2.resize(img, (w, h))
grouped_image = np.zeros((h, w * len(anchor_paths), 3), dtype=np.uint8)
anchor_keypoints = []
anchor_descriptors = []

# Get the keypoints and descriptors for the anchor images
for anc_index, anchor_image_path in enumerate(anchor_paths):
    print(f'Processing anchor image {anc_index + 1}/{len(anchor_paths)}')
    # First read the original image to get the mask using transparency
    backgroundless_img = cv2.imread("data\\anchor_no_bg\\anchor_image_"+str(anc_index+1)+".png", cv2.IMREAD_COLOR)
    img = cv2.imread(anchor_image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale for SIFT if the image has RGB channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    h = int(h * anchor_scaling)
    w = int(w * anchor_scaling)
    anchor_w = w
    anchor_h = h
    img = cv2.resize(img, (w, h))
    # Apply median blur to the image to remove noise and small details not present in target images
    img = median_blur(img, 3)
    # Apply Gaussian blur to the image to make image smoother, closer to the target images
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Resize backgroundless image to match the anchor image
    backgroundless_img = cv2.resize(backgroundless_img, (img.shape[1], img.shape[0]))
    mask, foreground = get_anchor_mask(backgroundless_img)
    kp, des = detect_sift_features(img)
    # Save the foreground image for debugging
    cv2.imwrite(f'output/anchor_foreground/{anchor_image_path.split("\\")[-1]}', foreground)

    # Get rid of any keypoints that are in the background
    filtered_kp = []
    filtered_des = []
    for i, keypoint in enumerate(kp):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        if mask[y, x] > 0:
            filtered_kp.append(keypoint)
            filtered_des.append(des[i])

    # Add the filtered keypoints and descriptors to the list
    anchor_keypoints.append(filtered_kp)
    anchor_descriptors.append(filtered_des)
    img = cv2.drawKeypoints(img, anchor_keypoints[anc_index], img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'output/anchor_features/anchor_{anc_index}.png', img)

    # Generate the grouped image
    img = cv2.imread(anchor_image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    h = int(h * anchor_scaling)
    w = int(w * anchor_scaling)

    # Resize the image to fit the grouped image
    img = cv2.resize(img, (w, h))
    # Use the mask to remove the background from the image
    mask = cv2.resize(mask, (w, h))
    img = img * (mask[:, :, None] > 0)
    grouped_image[:, anc_index * w:(anc_index + 1) * w] = img

print(f'Number of anchor images: {len(anchor_paths)}, starting SIFT feature detection on target images...')

# Save the grouped image to the output folder
cv2.imwrite('output/anchor_grouped.png', grouped_image)

# Using clustering, determine the colors that best represent the range of pixel values in the anchor images
# One of these colors will be the background. The other two will be the colors of the objects in the anchor images.
color_with_frequencies = []
img = grouped_image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 22
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image to show the image
# Center is the most important value, it stores the values of the 8 colors found by the k-meeans algorithm.
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
print(center)
print(ret)
print(label)

# Get the frequency of each color in the image
for anchor_color in center:
    # Ignore the black background color
    if(anchor_color[0] >= 2 and anchor_color[1] >= 2 and anchor_color[2] >= 2):
        count = np.count_nonzero(res == anchor_color)
        color_with_frequencies.append((anchor_color, count))

# Sort the colors by frequency
color_with_frequencies.sort(key=lambda x: x[1], reverse=True)

# Save the histogram of colors to the output folder
fig, ax = plt.subplots()
colors = [color[0] for color in color_with_frequencies]
frequencies = [color[1] for color in color_with_frequencies]
ax.bar(range(len(colors)), frequencies, color=[color / 255 for color in colors])
ax.set_xticks(range(len(colors)))
ax.set_xticklabels([f'({color[0]}, {color[1]}, {color[2]})' for color in colors])

# Get the top 2 colors (excluding the background color)
color_with_frequencies = color_with_frequencies[:2]

# Save the histogram to the output folder
plt.savefig('output/color_histogram.png')

res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

# Write a chart of the colors and their frequencies to the output folder
fig, ax = plt.subplots()
colors = [color[0] for color in color_with_frequencies]
frequencies = [color[1] for color in color_with_frequencies]
ax.bar(range(len(colors)), frequencies, color=[color / 255 for color in colors])
ax.set_xticks(range(len(colors)))
ax.set_xticklabels([f'({color[0]}, {color[1]}, {color[2]})' for color in colors])

# Get the target images
target_images = glob.glob(target_root + '/*.png')
img = cv2.imread(target_images[0])
h, w = img.shape[:2]
print(f'Number of target images: {len(target_images)}, pixel color distance for ROI determination on target images...')
print(f'Image dimensions: {h}x{w}')
count = 0

print (color_with_frequencies)

def get_rois_from_heatmap(heatmap, threshold):
    # Threshold the heatmap to create a binary image
    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    binary_map = binary_map.astype(np.uint8)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    output_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)  # Convert heatmap to color for visualization

    # Iterate through contours and extract bounding rectangles
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Apply padding to the bounding rectangle
        padded_w = int(w * (image_roi_padding[1] + image_roi_padding[3]))
        padded_h = int(h * (image_roi_padding[0] + image_roi_padding[2]))
        x = max(0, x - int(w * image_roi_padding[3]))
        y = max(0, y - int(h * image_roi_padding[0]))
        w = min(w + padded_w, heatmap.shape[1] - x)
        h = min(h + padded_h, heatmap.shape[0] - y)
        rois.append((x, y, w, h))
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0))  # Draw rectangles on ROIs
    
    return rois, output_img

# Process each target image
for target_path in target_images:
    count += 1
    print("Processing image", count)
    # Create a 2d array to represent the heat map of pixel interest
    img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    target_height = h
    target_width = w
    color_mapped_image = np.zeros((h, w, 3), np.uint8)
    heat_array = np.zeros((h, w), np.double)
    for i in range(h):
        for j in range(w):
            pixel = img[i, j]
            min_color_difference = np.int16(1000)
            intensity_difference = 0
            pixel_color = (pixel[0], pixel[1], pixel[2])
            freq = 0
            # Match the pixel color to the nearest anchor color
            for color_with_freq in color_with_frequencies:
                anchor_color = color_with_freq[0]
                # Get the difference in color channel values
                temp_intensity_diff = abs(np.int16(pixel[0]) + np.int16(pixel[1]) + np.int16(pixel[2]) - np.int16(anchor_color[0]) - np.int16(anchor_color[1]) - np.int16(anchor_color[2]))/3
                color_difference = abs(np.int16(pixel[0]) - np.int16(anchor_color[0]) - temp_intensity_diff) + abs(np.int16(pixel[1]) - np.int16(anchor_color[1]) - temp_intensity_diff) + abs(np.int16(pixel[2]) - np.int16(anchor_color[2]) - temp_intensity_diff)
                if min_color_difference > color_difference:
                    pixel_color = anchor_color
                    freq = color_with_freq[1]
                    min_color_difference = color_difference
                    intensity_difference = temp_intensity_diff
            # Calculate the interest of the pixel based on the color difference and the frequency of the color
            interest = 1 - min_color_difference / 765
            interest = interest * (1 - intensity_difference / 255)
            interest = interest * (freq//5000)
            heat_array[i, j] = interest
            color_mapped_image[i, j, 0] = pixel_color[2]
            color_mapped_image[i, j, 1] = pixel_color[1]
            color_mapped_image[i, j, 2] = pixel_color[0]

    if show_debug_images:
        # Get the first channel of the image
        img_channel_0 = img[:, :, 0]
        img_channel_1 = img[:, :, 1]
        img_channel_2 = img[:, :, 2]
        cv2.imwrite(f'output/example_img_channel_0.png', img_channel_0)
        cv2.imwrite(f'output/example_img_channel_1.png', img_channel_1)
        cv2.imwrite(f'output/example_img_channel_2.png', img_channel_2)
    # Save the distance array as an image

    # Apply a median blur to the heat array to smooth it out
    heat_array = median_blur(heat_array, 5)
    # Apply a Gaussian blur to the heat array to smooth it out
    heat_array = cv2.GaussianBlur(heat_array, (3, 3), 0)
    # Convert the heat array to 0-255
    heat_array = heat_array * 255 / np.max(heat_array)
    print(np.max(heat_array))
    print(np.min(heat_array))
    heat_image = heat_array.astype(np.uint8)
    cv2.imwrite(f'output/heat_images/{target_path.split("\\")[-1]}', heat_image)

    # Convert the heat array to a grayscale image
    heat_array = heat_array.astype(np.uint8)

    # Get the ROIs from the heat map
    # Set a threshold to define regions of interest
    threshold = max(70, np.median(heat_array)*2)

    # Get ROIs and visualization image
    rois, output_img = get_rois_from_heatmap(heat_array, threshold)

    # Save the output image with ROIs
    cv2.imwrite(f'output/roi_images/{target_path.split("\\")[-1]}', output_img)

    # Now that we have the ROIS, we can use the SIFT features to find the keypoints in the ROIs
    # and match them to the anchor image keypoints. We can then determine the the quality of each ROI match

    # For each ROI, get the sub-image and detect SIFT features
    
    best_roi_image = None
    print(f'Number of ROIs: {len(rois)}')
    best_matches = None
    best_match_count = 0
    best_anchor_id = 0
    best_roi = None
    best_match_quality = 0
    best_kp = None
    best_descriptors = None
    for roi in rois:
        # Load the target image
        target_image = cv2.imread(target_path, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        # Skip very small ROIs
        if (w < int(target_height * 0.03) or h < int(target_height * 0.1)):
            continue
        # Get the region of interest (ROI) from the target image
        roi_image = target_image[y:y + h, x:x + w]
        # Write the ROI image to a file for debugging
        if show_debug_images:
            cv2.imwrite(f'output/ROI.png', roi_image)
        kp, des = detect_sift_features(roi_image)
        print(f'Number of keypoints detected in ROI: {len(kp)}')
        if(len(kp) == 0):
            continue
        if(len(des) == 0):
            continue
        roi_best_matches = None

        # Match the SIFT features of the ROI to the anchor images
        # To ensure only one match per keypoint in the target image, we will limit the number of matches to 1
        for anchor_id, anchor_des in enumerate(anchor_descriptors):
            similarity_score = 0
            match_quality = float(0)
            match_count = 0
            matches, matches_list = match_sift_features(np.array([descriptor for descriptor in des]), np.array(anchor_des))
            # Determine the quality of the matches as a ratio of the number of matches to the number of matched keypoints in the anchor image
            if len(anchor_des) > 0:
                for match in matches:
                    if(match != None):
                        similarity_score = len(match) / len(anchor_des)
                        print(match)
                        # Get the size of the anchor keypoint 
                        keypoint_size = anchor_keypoints[anchor_id][match[0]].size
                        match_quality += float(1) / match[2]
                        match_quality /= float(len(anchor_des))
                match_count = len(matches)

            # Update the best ROI if the current ROI has more matches
            if match_quality > best_match_quality:
                best_match_count = match_count
                best_roi = roi
                best_matches = matches_list
                best_roi_image = roi_image
                best_anchor_id = anchor_id
                best_match_quality = match_quality
                best_kp = kp
                best_descriptors = des
                best_similarity_score = similarity_score
            
    # Draw the Best ROI on the target image
    if best_roi is not None:
        x, y, w, h = best_roi
        cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(f'output/target_with_identification/{target_path.split("\\")[-1]}', target_image)
        print(f'Best ROI found with {best_match_count} matches')
        print(f'Anchor image index: {best_anchor_id}')
        print(f'Match quality: {best_match_quality:.2f}')
    else:
        print('No ROI found')

    # Generate an image with the anchor image and the target image side by side with matches
    anchor_image = cv2.imread(anchor_paths[best_anchor_id], cv2.IMREAD_COLOR)
    anchor_image = cv2.resize(anchor_image, (int(len(anchor_image[0])*anchor_scaling), int(len(anchor_image)*anchor_scaling)))

    # Draw the matches between the anchor and target images
    print(best_matches)
    # Get the matches using the sift matching algorithm
    # Print the size of the anchor image
    
    if(best_matches == None or len(best_matches) == 0 or best_roi_image is None):
        continue
    
    # Draw the matches between the anchor and target image
    # Draw lines between the matches in the target image
    # Draw circles around the matches in the anchor image
    # Draw the images side by side, gaps are black (0)
    # The target image is on the left, the anchor image is on the right

    match_image = np.zeros((max(len(best_roi_image) , len(anchor_image)), len(best_roi_image[0]) + len(anchor_image[0])), np.uint8)
    match_image.fill(0)
    best_roi_image = np.uint8(best_roi_image)
    # Get the anchor_image from the average of its channels
    anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    anchor_image = np.uint8(anchor_image)
    # Draw the target image on the left
    match_image[:len(best_roi_image), :len(best_roi_image[0])] = best_roi_image
    # Draw the anchor image on the right
    match_image[:len(anchor_image), len(best_roi_image[0]):] = anchor_image
    # Draw the matches
    # Conver matches to an image with 3 channels
    match_image = cv2.cvtColor(match_image, cv2.COLOR_GRAY2BGR)
    for match in best_matches:
        # Get the keypoints
        kp1 = best_kp[match[0][0]]
        kp2 = anchor_keypoints[best_anchor_id][match[0][1]]
        # Get the coordinates of the keypoints
        x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
        x2, y2 = int(kp2.pt[0]), int(kp2.pt[1])
        # Draw a line between the keypoints
        cv2.line(match_image, (x1, y1), (x2 + len(best_roi_image[0]), y2), (0,255,0), 1)
        # Draw a circle around the keypoints
        cv2.circle(match_image, (x1, y1), 3, (255,0,0), 1)
        cv2.circle(match_image, (x2 + len(best_roi_image[0]), y2), 3, (0,0,255), 1)


    # Save the match image to the output folder
    cv2.imwrite(f'output/matches/{target_path.split("\\")[-1]}', match_image)

    # Save the target image with the best ROI identified
    target_image = cv2.imread(target_path, cv2.IMREAD_COLOR)
    x, y, w, h = best_roi
    cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Add test with the similarity score to the image
    cv2.putText(target_image, f"Similarity Score (out of 1): {best_match_quality:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(f'output\\target_with_identification\\{target_path.split("\\")[-1]}', target_image)


