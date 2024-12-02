import os
import numpy as np
from helper_function.bbhelp import extract_image_info

labels_file_path = r'C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data\reect\seq_02\labels.txt'
images_folder_path = r'C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data\reect\seq_02\image_02\data'
depth_maps_path = r'C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data\sgbm_depth_maps\seq_02'

P_rect_02 = np.array([
    [7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
    [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
    [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]
])

K = P_rect_02[:3, :3]
K_inv = np.linalg.inv(K)

image_files = sorted(os.listdir(images_folder_path))
predictions = []

for i, image_file in enumerate(image_files):
    labels = extract_image_info(i,labels_file_path)
    print(f'read labels for image {i:06d}')
    depth = np.load(os.path.join(depth_maps_path, f'{i:010d}.npy'))
    preds = labels.copy()

    for pred in preds:
        center = np.array([pred['center'][0], pred['center'][1], 1])
        position_stereo = depth[int(pred['center'][1]), int(pred['center'][0])] * (K_inv @ center)
        pred['pos'] = position_stereo

    for pred in preds:
            predictions.append(
                f"{i} {pred['track_id']} {pred['type']} {pred['bbox'][0]} {pred['bbox'][1]} {pred['bbox'][2]} {pred['bbox'][3]} "
                f"{pred['dimensions'][0]} {pred['dimensions'][1]} {pred['dimensions'][2]} "
                f"{pred['location'][0]} {pred['location'][1]} {pred['location'][2]} {pred['rotation_y']}\n"
            )

preds_file = "stereo_preds_seq_02_all_.txt"
with open(preds_file, 'w') as f:
    f.writelines(predictions)
