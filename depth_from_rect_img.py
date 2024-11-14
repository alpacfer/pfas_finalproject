import cv2
import numpy as np
import os
import glob

def generate_depth_maps(base_dir, sequence):

    # Paths to the directories containing the rectified images
    left_image_dir = os.path.join(base_dir, sequence, "image_02", "data")
    right_image_dir = os.path.join(base_dir, sequence, "image_03", "data")
    output_dir = os.path.join(base_dir, "sgbm_depth_maps", sequence)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the directories
    left_images = sorted(glob.glob(os.path.join(left_image_dir, '*.png')))
    right_images = sorted(glob.glob(os.path.join(right_image_dir, '*.png')))

    # Check if the number of left and right images are the same
    if len(left_images) != len(right_images):
        print("Error: The number of left and right images do not match")
        return

    # Create StereoBM or StereoSGBM object
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=3)

    # Process each pair of images
    for left_image_path, right_image_path in zip(left_images, right_images):
        # Load the images
        left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded properly
        if left_image is None or right_image is None:
            print(f"Error: Could not load images {left_image_path} and {right_image_path}")
            continue

        # Compute the disparity map
        disparity = stereo.compute(left_image, right_image)

        # Normalize the disparity map for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        # Generate output file path
        output_file_path = os.path.join(output_dir, os.path.basename(left_image_path))

        # Save the disparity map
        cv2.imwrite(output_file_path, disparity_normalized)

        print(f"Saved disparity map to {output_file_path}")

    print("Processing complete.")

# Example usage
base_dir = r"path"
generate_depth_maps(base_dir, "seq_01")
generate_depth_maps(base_dir, "seq_02")
generate_depth_maps(base_dir, "seq_03")
