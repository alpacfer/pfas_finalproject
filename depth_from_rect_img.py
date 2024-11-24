import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def preprocess_image(image):
    """Apply histogram equalization to improve contrast."""
    return cv2.equalizeHist(image)


def postprocess_disparity(disparity):
    """Refine disparity map using a Weighted Median Filter."""
    # Convert disparity to 8-bit format
    disparity_8u = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply median blur
    kernel_size = 9  # Filter size for refinement
    refined_disparity = cv2.medianBlur(disparity_8u, kernel_size)

    # Convert back to 32-bit float for consistency
    refined_disparity = refined_disparity.astype(np.float32)
    return refined_disparity

def get_non_overlapping_region(disparity, min_disparity, max_disparity, width):
    """
    Given the disparity map, min_disparity, max_disparity, and width of the image,
    this function returns the indices for the non-overlapping regions in both
    left and right images.

    :param disparity: Disparity map
    :param min_disparity: Minimum disparity value (usually 0)
    :param max_disparity: Maximum disparity value
    :param width: Width of the image
    :return: Binary masks for non-overlapping regions
    """
    # Create mask for valid disparity (within min and max disparity range)
    valid_mask = (disparity >= min_disparity) & (disparity <= max_disparity)

    # Identify the overlap region (valid disparity region)
    overlap_start = np.argmax(valid_mask[0, :])  # Find first column with valid disparity
    overlap_end = np.argmax(valid_mask[0, ::-1])  # Find last column with valid disparity

    # Compute the non-overlapping regions
    left_non_overlap = np.ones_like(valid_mask)
    left_non_overlap[:, :overlap_start] = 0  # Non-overlapping part of the left image

    right_non_overlap = np.ones_like(valid_mask)
    right_non_overlap[:, overlap_end:] = 0  # Non-overlapping part of the right image

    return left_non_overlap, right_non_overlap


def generate_depth_maps(base_dir, sequence):
    # Paths to the directories containing the rectified images
    left_image_dir = os.path.join(base_dir, sequence, "image_02", "data")
    right_image_dir = os.path.join(base_dir, sequence, "image_03", "data")
    output_dir = os.path.join(base_dir, "sgbm_depth_maps", sequence)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the directories
    left_images = sorted(glob.glob(os.path.join(left_image_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_image_dir, "*.png")))

    if len(left_images) != len(right_images):
        print("Error: The number of left and right images do not match")
        return

    # SGBM parameters based on literature for KITTI
    min_disp = 0
    num_disp = 192  # Must be divisible by 16
    block_size = 9
    uniqueness_ratio = 3
    speckle_window_size = 100
    speckle_range = 2
    disp_max_diff = 1
    p1 = 8 * 3 * block_size ** 2
    p2 = 32 * 3 * block_size ** 2

    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Process each image pair
    for left_image_path, right_image_path in zip(left_images, right_images):
        left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        if left_image is None or right_image is None:
            print(f"Error: Could not load images {left_image_path} and {right_image_path}")
            continue

        # Preprocessing
        left_image = preprocess_image(left_image)
        right_image = preprocess_image(right_image)

        # Compute disparity map
        disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        non_overlap = get_non_overlapping_region(disparity, min_disp, min_disp + num_disp, left_image.shape[1])

        # Postprocessing
        disparity = postprocess_disparity(disparity)

        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#712704
        disparity_normalized = np.uint8(disparity_normalized)

        # Save results
        output_file_path = os.path.join(output_dir, os.path.basename(left_image_path))
        cv2.imwrite(output_file_path, disparity_normalized)

        #Debugging: Coverage and visualization
        coverage = np.count_nonzero(disparity_normalized[non_overlap[0] == 1]) / np.count_nonzero(non_overlap[0])
        print(f"Saved disparity map to {output_file_path}")
        print(f'Img Shape: {left_image.shape}')
        print(f'Disparity Map Shape {disparity.shape}')
        print(f'disparity Map size: {disparity_normalized.size}')
        print(f'non-overlapping region size: {np.count_nonzero(non_overlap[0])}')
        print(f'disparity Map non-overlapping region size: {np.count_nonzero(disparity_normalized[non_overlap[0] == 1])}')
        print(f"Coverage: {coverage * 100:.2f}%")

        # plt.figure(figsize=(10, 5))
        fig,ax =plt.subplots(1, 2)
        ax[0].imshow(left_image, cmap="gray")
        # Create a Rectangle: (x, y) is the bottom-left corner, width, height
        # rect = patches.Rectangle((0, 0), n, 40, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axis
        # ax[0].add_patch(rect)
        
        ax[0].set_title("Left Image")
        ax[0].axis("off")
        ax[1].imshow(disparity_normalized, cmap="plasma")
        ax[1].set_title("Disparity Map")
        ax[1].axis("off")

        # ax[2].imshow(non_overlap[0], cmap="gray")
        # ax[2].set_title("Non-overlapping Region Mask")
        plt.show()

    print("Processing complete.")


# Example usage
base_dir = r"C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data"
generate_depth_maps(base_dir, "seq_01")
generate_depth_maps(base_dir, "seq_02")
generate_depth_maps(base_dir, "seq_03")
