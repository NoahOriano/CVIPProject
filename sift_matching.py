import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


# Define paths to the anchor and target image folders
anchor_root = 'data/anchor_no_bg'
target_root = 'data/camera_color_image_raw'

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect the SIFT features for the anchor images
anchor_images = glob.glob(anchor_root + '/*.png')
anchor_images.sort()
anchor_keypoints = []
anchor_descriptors = []

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
        _, mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)

    # Apply the mask to focus on the object
    foreground = cv2.bitwise_and(gray, gray, mask=mask)
    return foreground, mask

# Get the keypoints and descriptors for the anchor images
for anchor_image_path in anchor_images:
    # First read the original image to get the mask using transparency
    img = cv2.imread(anchor_image_path, cv2.IMREAD_UNCHANGED)
    mask, foreground = get_anchor_mask(img)
    kp, des = sift.detectAndCompute(foreground, None)
    anchor_keypoints.append(kp)
    anchor_descriptors.append(des)

print(f'Number of anchor images: {len(anchor_images)}, starting SIFT feature detection on target images...')


# Helper function to detect SIFT features for the target images
def detect_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

# Display the SIFT features and their orrientation for each of the anchor images and save them
for i, anchor_image_path in enumerate(anchor_images):
    img = cv2.imread(anchor_image_path)
    img = cv2.drawKeypoints(img, anchor_keypoints[i], img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'output/anchor_features/anchor_{i}.png', img)

# Create a grouped image to show all the anchor images together
h, w = img.shape[:2]
grouped_image = np.zeros((h, w * len(anchor_images), 3), dtype=np.uint8)
for i, anchor_image_path in enumerate(anchor_images):
    img = cv2.imread(anchor_image_path)
    grouped_image[:, i * w:(i + 1) * w] = img

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
K = 9
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

# Remove the least frequent colors
color_with_frequencies = color_with_frequencies[:3]

# Write the reduced color image to the output folder
cv2.imwrite(f'output/anchor_reduced_colors/anchors.png', res2)

# Write a chart of the colors and their frequencies to the output folder
fig, ax = plt.subplots()
colors = [color[0] for color in color_with_frequencies]
frequencies = [color[1] for color in color_with_frequencies]
ax.bar(range(len(colors)), frequencies, color=[color / 255 for color in colors])
ax.set_xticks(range(len(colors)))
ax.set_xticklabels([f'({color[0]}, {color[1]}, {color[2]})' for color in colors])
plt.savefig('output/anchor_colors.png')

# Now, we have the sift features and the expected colors of the anchor image.
# We can now start by finding regions of interest by finding the uclidean distance
# between the anchor colors and the target image colors.
# To do this, we will first convert the image into superpixels. Then, we will find the average color of each superpixel
# The distance between the superpixel color and the nearest anchor image color will be determined. Then, each superpixel that is
# Near enough will be considered part of a region of interest, connected superpixels will be considered to determine an enclosing rectangle
# around the region of interest.

# Get the target images
target_images = glob.glob(target_root + '/*.png')
img = cv2.imread(target_images[0])
h, w = img.shape[:2]
print(f'Number of target images: {len(target_images)}, pixel color distance for ROI determination on target images...')
print(f'Image dimensions: {h}x{w}')
count = 0

print (color_with_frequencies)


# Process each target image
for target_path in target_images:
    count += 1
    print("Processing image", count)
    # Create a 2d array to represent the heat map of pixel interest
    img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                temp_intensity_diff = abs(np.int16(pixel[0]) + np.int16(pixel[1]) + np.int16(pixel[2]) - np.int16(anchor_color[0]) - np.int16(anchor_color[1]) - np.int16(anchor_color[2]))/6
                color_difference = abs(np.int16(pixel[0]) - np.int16(anchor_color[0]) - temp_intensity_diff) + abs(np.int16(pixel[1]) - np.int16(anchor_color[1]) - temp_intensity_diff) + abs(np.int16(pixel[2]) - np.int16(anchor_color[2]) - temp_intensity_diff)
                if min_color_difference > color_difference:
                    pixel_color = anchor_color
                    freq = color_with_freq[1]
                    min_color_difference = color_difference
                    intensity_difference = temp_intensity_diff
            # Calculate the interest/heat of the pixel
            interest = freq / (min_color_difference + 5 + intensity_difference)
            heat_array[i, j] = interest
            color_mapped_image[i, j, 0] = pixel_color[2]
            color_mapped_image[i, j, 1] = pixel_color[1]
            color_mapped_image[i, j, 2] = pixel_color[0]
        
    # Apply a median blur to the heat array to smooth it out
    heat_array = median_blur(heat_array, 5)

    # Save the color-mapped image
    cv2.imwrite(f'output/example_mapping.png', color_mapped_image)
    # Get the first channel of the image
    img_channel_0 = img[:, :, 0]
    img_channel_1 = img[:, :, 1]
    img_channel_2 = img[:, :, 2]
    cv2.imwrite(f'output/example_img_channel_0.png', img_channel_0)
    cv2.imwrite(f'output/example_img_channel_1.png', img_channel_1)
    cv2.imwrite(f'output/example_img_channel_2.png', img_channel_2)
    # Save the distance array as an image
    # Normalize the heat array to 0-255
    heat_array = heat_array * 255 / np.max(heat_array)
    print(np.max(heat_array))
    print(np.min(heat_array))
    heat_image = heat_array.astype(np.uint8)
    cv2.imwrite(f'output/heat_images/{target_path.split("\\")[-1]}', heat_image)



