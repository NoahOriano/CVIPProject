import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import glob

# Status: works as expected
def generate_scale_space(image, num_octaves=4, num_scales=5, initial_sigma=1.6):
    octaves = []
    for octave in range(num_octaves):
        scales = []
        sigma = initial_sigma * (2 ** octave)
        for scale in range(num_scales):
            blurred = gaussian_filter(image, sigma=sigma)
            scales.append(blurred)
            sigma *= np.sqrt(2)  # Increase sigma for next scale
        octaves.append(scales)
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))  # Downscale for next octave
    return octaves

# Status: works as expected
def generate_dog_octaves(octaves):
    dog_octaves = []
    for scales in octaves:
        dog = []
        for i in range(1, len(scales)):
            dog.append(scales[i] - scales[i - 1])
        dog_octaves.append(dog)
    return dog_octaves

# Status: does not work as expected, can't figure out how to correctly get patch since s is a tuple
def find_keypoints(dog_octaves, threshold=0.03):
    keypoints = []
    for o, octave in enumerate(dog_octaves):
        for s in range(1, len(octave) - 1):
            for y in range(1, octave[s].shape[0] - 1):
                for x in range(1, octave[s].shape[1] - 1):
                    patch = octave[s, y - 1:y + 2, x - 1:x + 2]
                    if is_extremum(patch, octave[s][y, x], threshold):
                        keypoints.append((o, s, x, y))
    return keypoints

# Status: duh it works its p basic
def is_extremum(patch, center_value, threshold):
    # Check if center pixel is a local extremum
    return (center_value > np.max(patch) - threshold or center_value < np.min(patch) + threshold)


def assign_orientation(image, keypoints, num_bins=36):
    orientations = []
    for kp in keypoints:
        x, y = kp[2], kp[3]
        magnitude, angle = compute_gradient(image, x, y)
        orientation_histogram = np.zeros(num_bins)
        bin_width = 360 // num_bins
        for m, a in zip(magnitude, angle):
            bin_index = int(a / bin_width) % num_bins
            orientation_histogram[bin_index] += m
        dominant_orientation = np.argmax(orientation_histogram)
        orientations.append((kp, dominant_orientation * bin_width))
    return orientations

def compute_gradient(image, x, y, size=3):
    gx = cv2.Sobel(image[y-size:y+size+1, x-size:x+size+1], cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image[y-size:y+size+1, x-size:x+size+1], cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 360
    return magnitude, angle

def compute_descriptors(image, orientations, window_size=16, num_bins=8):
    descriptors = []
    for kp, orientation in orientations:
        x, y = kp[2], kp[3]
        descriptor = np.zeros((num_bins,))
        magnitude, angle = compute_gradient(image, x, y, window_size//2)
        bin_width = 360 // num_bins
        for m, a in zip(magnitude, angle):
            relative_angle = (a - orientation + 360) % 360
            bin_index = int(relative_angle / bin_width) % num_bins
            descriptor[bin_index] += m
        descriptors.append(descriptor / np.linalg.norm(descriptor))  # Normalize for invariance
    return descriptors

# Used to run the entire SIFT pipeline on an image, returning the keypoints and descriptors
# Note only the orientations and descriptors are needed, keypoints is just there for debugging
def sift_pipeline(image):
    # Step 1: Scale-Space
    octaves = generate_scale_space(image)
    # Step 2: Difference of Gaussian
    dog_octaves = generate_dog_octaves(octaves)
    # Step 3: Keypoint Localization
    keypoints = find_keypoints(dog_octaves)
    # Step 4: Orientation Assignment
    orientations = assign_orientation(image, keypoints)
    # Step 5: Descriptor Calculation
    descriptors = compute_descriptors(image, orientations)
    return orientations, descriptors, keypoints

# Read each anchor image and then process the sift features
anchor_image_names = glob.glob("data/anchor_no_bg/*.png")
anchor_images = [cv2.imread(img) for img in anchor_image_names]
for img in anchor_images:
    orientations, descriptors = sift_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print("Number of Keypoints Detected:", len(orientations))
    print("Descriptors Shape:", np.shape(descriptors))
    print("First Descriptor:", descriptors[0])
    print("First Orientation:", orientations[0])

        

        
