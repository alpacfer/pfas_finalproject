from cProfile import label
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import gt
from helper_function.bbhelp import extract_image_info
import torch


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

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def to_depth(prediction, target, mask,depth_cap=80):
    # transform predicted disparity to aligned depth
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    prediciton_depth = 1.0 / prediction_aligned

    return prediciton_depth


def generate_depth_maps(base_dir, sequence):
    # Paths to the directories containing the rectified images
    left_image_dir = os.path.join(base_dir, "reect",sequence, "image_02", "data")
    right_image_dir = os.path.join(base_dir, "reect",sequence, "image_03", "data")
    output_dir = os.path.join(base_dir, "sgbm_depth_maps", sequence)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the directories
    left_images = sorted(glob.glob(os.path.join(left_image_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_image_dir, "*.png")))

    if len(left_images) != len(right_images):
        print("Error: The number of left and right images do not match")
        return
    
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # SGBM parameters based on literature for KITTI
    min_disp = 0
    num_disp = 16 * 20  # Must be divisible by 16
    block_size = 5
    uniqueness_ratio = 1
    speckle_window_size = 3
    speckle_range = 1
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

    # stereo = cv2.StereoBM_create(
    #     numDisparities=num_disp, 
    #     blockSize=block_size)
    
    # # Set additional fine-tuning parameters
    # stereo.setMinDisparity(min_disp)  # Sets the minimum disparity
    # stereo.setDisp12MaxDiff(disp_max_diff)  # Max difference between left and right disparities. Lower = more accurate, higher = looser.
    # stereo.setUniquenessRatio(uniqueness_ratio)  # How unique a match must be. Higher = stricter matching, lower = more noise.
    # stereo.setSpeckleRange(speckle_range)  # Max disparity variation in speckles (noise). Lower = more aggressive noise removal.
    # stereo.setSpeckleWindowSize(speckle_window_size)  # Size of noisy regions to be removed. Higher = keep more valid data but may retain noise.


    # Parameters
    # minDisparity	Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    # numDisparities	Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
    # blockSize	Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    # P1	The first parameter controlling the disparity smoothness. See below.
    # P2	The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively).
    # disp12MaxDiff	Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    # preFilterCap	Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
    # uniquenessRatio	Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    # speckleWindowSize	Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    # speckleRange	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    # mode	Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .

    # Process each image pair
    for id,image_pair in enumerate(zip(left_images, right_images)):
        left_image_path, right_image_path = image_pair
        print(f'Generating Depth: {id+1}/{len(left_images)}')
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        if left_image is None or right_image is None:
            print(f"Error: Could not load images {left_image_path} and {right_image_path}")
            continue

        

        # Preprocessing
        left_gray = preprocess_image(left_gray)
        right_gray = preprocess_image(right_gray)



        # Compute disparity map
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        non_overlap = get_non_overlapping_region(disparity, min_disp, min_disp + num_disp, left_gray.shape[1])

        # Postprocessing
        disparity = postprocess_disparity(disparity)
        # Convert disparity to depth
        focal_length = 707  # Example focal length, replace with actual value
        baseline = 0.54  # Example baseline distance in meters, replace with actual value
        depth = (focal_length * baseline) / (disparity + 1e-6)  # Add a small value to avoid division by zero
        depth [depth>80]=0

        # Dense Depth Prediction from MiDaS calibrated with Metric Measurements from Stereo
        input_batch = transform(left_image)
        with torch.no_grad():
            pred_depth = midas(input_batch)

            pred_depth = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(1),
                size=left_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        
        pred_depth = pred_depth.cpu().numpy()

        mask = (0 < depth ) & (depth < 80)
        pred_depth_metric = to_depth(torch.tensor(pred_depth).unsqueeze(0), torch.tensor(depth).unsqueeze(0), torch.tensor(mask).unsqueeze(0))



       

        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#712704
        disparity_normalized = np.uint8(disparity_normalized)

        # Save results
        output_file_path = os.path.join(output_dir, os.path.basename(left_image_path))
        np.save(output_file_path[:-4], depth)
        np.save(f"{output_file_path[:-4]}_midas", pred_depth)
        np.save(f"{output_file_path[:-4]}_midas_metric", pred_depth_metric)
        cv2.imwrite(output_file_path, disparity_normalized)

        #Debugging: Coverage and visualization
        coverage = np.count_nonzero(disparity_normalized[non_overlap[0] == 1]) / np.count_nonzero(non_overlap[0])
        viz = True
        if viz:
            print(f"Saved disparity map to {output_file_path}")
            print(f'Img Shape: {left_image.shape}')
            print(f'Disparity Map Shape {disparity.shape}')
            print(f'disparity Map size: {disparity_normalized.size}')
            print(f'non-overlapping region size: {np.count_nonzero(non_overlap[0])}')
            print(f'disparity Map non-overlapping region size: {np.count_nonzero(disparity_normalized[non_overlap[0] == 1])}')
            print(f"Coverage: {coverage * 100:.2f}%")

            # plt.figure(figsize=(10, 5))
            fig,ax =plt.subplots(1, 3)
            ax[0].imshow(left_image, cmap="gray")
            # Create a Rectangle: (x, y) is the bottom-left corner, width, height
            # rect = patches.Rectangle((0, 0), n, 40, linewidth=2, edgecolor='r', facecolor='none')

            # Add the rectangle to the axis
            # ax[0].add_patch(rect)
            
            ax[0].set_title("Left Image")
            ax[0].axis("off")
            ax[1].imshow(depth, cmap="plasma")
            ax[1].set_title("Disparity Map")
            ax[1].axis("off")

            # ax[2].imshow(non_overlap[0], cmap="gray")
            # ax[2].set_title("Non-overlapping Region Mask")
            ax[2].imshow(pred_depth_metric[0], cmap="gray")
            plt.show()

    print("Processing complete.")




def evaluate_sequence_depth(base_dir,sequence_id=1):
    P2 = np.array([[707.0493, 0, 604.0814, 45.75831],
                  [0, 707.0493, 180.5066, -0.3454157],
                  [0, 0, 1, 0.004981016]])
    K2 = P2[:3,:3]

    img_dir = os.path.join(base_dir, "reect",f"seq_0{sequence_id}", "image_02", "data")
    depth_dir = os.path.join(base_dir, "sgbm_depth_maps",f"seq_0{sequence_id}")

    labels_file = os.path.join(base_dir,f"seq_0{sequence_id}","labels.txt")

    n_frames = len(os.listdir(img_dir))

    seq_gt_stereo = []
    seq_errors_stereo = []
    seq_gt_midas = []
    seq_errors_midas = []
    for i in range(n_frames):
        labels = extract_image_info(i,labels_file)
        depth_stereo = np.load(os.path.join(depth_dir,f"{i:06}.npy"))
        depth_midas = np.load(os.path.join(depth_dir,f"{i:06}_midas_metric.npy"))[0]
        img = cv2.imread(os.path.join(img_dir,f"{i:06}.png"))

        assert depth_stereo.shape == depth_midas.shape ==img.shape[:2], 'Depth and Image shape mismatch'

        

        # fig,ax =plt.subplots(1, 2)
        for obj in labels:
            if obj['occluded']!=0:
                continue

            obj_center = K2 @ (np.array(obj['location']) + np.array([0,-obj['dimensions'][0]/2,0])).reshape(-1,1)
            obj_center = obj_center[:2]/obj_center[2]
            obj_center = obj_center.reshape(-1)
            # ax[0].scatter(int(obj_center[0]),int(obj_center[1]),c='r')

            if obj_center[0]<0 or obj_center[0]>=img.shape[1] or obj_center[1]<0 or obj_center[1]>=img.shape[0]:
                continue

            if depth_stereo[int(obj_center[1]),int(obj_center[0])]!=0:
                print(f"Pred Depth: {depth_stereo[int(obj_center[1]),int(obj_center[0])]} VS GT Depth: {obj['location'][2]}")
                seq_gt_stereo .append(obj['location'][2])
                seq_errors_stereo.append(abs(depth_stereo[int(obj_center[1]),int(obj_center[0])]-obj['location'][2]))

            if depth_midas[int(obj_center[1]),int(obj_center[0])]!=0:
                seq_gt_midas.append(obj['location'][2])
                seq_errors_midas.append(abs(depth_midas[int(obj_center[1]),int(obj_center[0])]-obj['location'][2]))




        
        # ax[0].imshow(img[:,:,::-1], cmap="gray")
        # # Create a Rectangle: (x, y) is the bottom-left corner, width, height
        # # rect = patches.Rectangle((0, 0), n, 40, linewidth=2, edgecolor='r', facecolor='none')

        # # Add the rectangle to the axis
        # # ax[0].add_patch(rect)
        
        # ax[0].set_title("Left Image")
        # ax[0].axis("off")
        # ax[1].imshow(depth, cmap="gray")
        # ax[1].set_title("Depth Map")
        # ax[1].axis("off")
        # plt.show()

        # ax[2].imshow(non_overlap[0], cmap="gray")
        # ax[2].set_title("Non-overlapping Region Mask")
    plt.scatter(seq_gt_stereo,seq_errors_stereo,label='Stereo')
    plt.scatter(seq_gt_midas,seq_errors_midas,label='MiDaS')
    plt.title(f"Sequence 0{sequence_id} Depth Error Distribution \n Stereo {len(seq_errors_stereo)} Objects \n Stereo {len(seq_errors_midas)} Objects \n Stereo AbsRel Error: {np.mean(np.array(seq_errors_stereo)/np.array(seq_gt_stereo))} \n Stereo AbsRel Error: {np.mean(np.array(seq_errors_midas)/np.array(seq_gt_midas))}")
    plt.xlabel("Ground Truth Depth [m]")
    plt.ylabel("Depth Error [m]")
    # plt.xlim(0,len(seq_errors)+1)
    plt.legend()
    plt.show()

    

# Example usage
base_dir = r"C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data"
# generate_depth_maps(base_dir, "seq_01")
# generate_depth_maps(base_dir, "seq_02")
# generate_depth_maps(base_dir, "seq_03")

evaluate_sequence_depth(base_dir,1)
