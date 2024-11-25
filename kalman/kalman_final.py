import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define the file paths
labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\helper_function\obstructed_labels'
images_folder_path = r'C:\Users\User\Documents\dataset\pfas\34759_final_project_rect\seq_02\image_02\data'

# Read the labels.txt file
headers = [
    "frame", "track id", "type", "truncated", "occluded", "alpha",
    "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
    "dimensions_height", "dimensions_width", "dimensions_length",
    "location_x", "location_y", "location_z",
    "rotation_y", "score"
]
df = pd.read_csv(labels_file_path, sep=' ', names=headers)

# Define constants
dt = 0.1036  # Time difference between frames
depth = 0  # Assumed constant depth for this case
max_distance = 50  # Maximum distance for matching
max_missing_frames = 5  # Max frames an object can be missing

# Initialize variables
missing_ids = []          # IDs of objects missing in current frame
tracked_ids = []          # IDs of objects being tracked
new_ids = []              # IDs of new detections in current frame
locations_dict = {}       # Stores bounding boxes for each track_id
previous_frame_ids = []   # IDs detected in previous frame
previous_frame_locations = {}  # Bounding boxes from previous frame
frame_counter = 1         # Counter for the current frame
reassociation_map = {}    # Maps new_ids to missing_ids
kalman_filters = {}       # Kalman filter instances for each track_id
missing_counts = {}       # Counts how long each missing_id has been missing

# Get sorted list of frames
unique_frames = sorted(df["frame"].unique())

# Kalman Filter Class Definition (using your functions)
class KalmanFilter:
    def __init__(self, initial_state):
        self.state_dim = 9  # x, y, z, vx, vy, vz, ax, ay, az
        self.meas_dim = 3  # x, y, z
        self.dt = dt
        self.depth = depth

        # Initial state vector
        self.x = np.zeros((self.state_dim, 1))
        self.x[:3, 0] = initial_state  # Initialize position

        # Transition matrix
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0, 0],
            [0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0],
            [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2],
            [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0, 0]   # z
        ])

        # Process covariance
        self.Q = np.eye(self.state_dim) * 0.0  # Adjusted to trust the system model more

        # Measurement covariance
        self.R = np.eye(self.meas_dim) * 0.01  # Adjusted to trust the measurements more

        # Initial covariance matrix
        self.P = np.eye(self.state_dim) * 1000

        # Flag to indicate if the filter was updated with a measurement
        self.updated_with_measurement = False

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.updated_with_measurement = False  # Reset flag during prediction

    def update(self, measurement):
        z = np.array(measurement).reshape(-1, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        self.updated_with_measurement = True  # Set flag when updated with measurement

    def get_current_state(self):
        return self.x

    def get_predicted_location(self):
        return self.x[:2, 0]  # Return x and y positions

def get_frame_data(frame_id):
    frame_data = df[df["frame"] == frame_id]
    frame_ids = frame_data["track id"].unique().tolist()
    frame_locations = {}
    for _, row in frame_data.iterrows():
        track_id = row["track id"]
        bbox = [row["bbox_left"], row["bbox_top"], row["bbox_right"], row["bbox_bottom"]]
        frame_locations[track_id] = bbox
    return frame_ids, frame_locations

def track_states(current_frame_ids, previous_frame_ids):
    missing_ids = [id_ for id_ in previous_frame_ids if id_ not in current_frame_ids]
    tracked_ids = [id_ for id_ in current_frame_ids if id_ in previous_frame_ids]
    new_ids = [id_ for id_ in current_frame_ids if id_ not in previous_frame_ids]
    return missing_ids, tracked_ids, new_ids

def compute_distance(bbox1, bbox2):
    # Compute Euclidean distance between the centers of two bounding boxes
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def match_new_ids_to_missing_predictions(new_ids, missing_ids_prediction, current_frame_locations, max_distance):
    matched_ids = []
    reassociation_map = {}
    unmatched_new_ids = []

    for new_id in new_ids:
        min_distance = float('inf')
        best_match_id = None
        new_bbox = current_frame_locations[new_id]

        for missing_id, predicted_bbox in missing_ids_prediction.items():
            distance = compute_distance(new_bbox, predicted_bbox)
            if distance < min_distance:
                min_distance = distance
                best_match_id = missing_id

        if min_distance < max_distance:
            matched_ids.append((best_match_id, new_id))
            reassociation_map[new_id] = best_match_id
            # Remove matched missing_id from missing_ids_prediction
            missing_ids_prediction.pop(best_match_id)
        else:
            unmatched_new_ids.append(new_id)

    return matched_ids, reassociation_map, unmatched_new_ids

def extract_bbox_from_state(state_tl, state_br):
    x_top_left = state_tl[0, 0]
    y_top_left = state_tl[1, 0]
    x_bottom_right = state_br[0, 0]
    y_bottom_right = state_br[1, 0]
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]

def is_bbox_out_of_frame(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    return (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > frame_height)

# Initialize lists to keep track of updates for visualization
updated_ids = []
predicted_only_ids = []
about_to_remove_ids = []

# Main processing loop
for frame in unique_frames:
    # Clear update tracking lists
    updated_ids.clear()
    predicted_only_ids.clear()
    about_to_remove_ids.clear()

    # Step 2: Retrieve Frame Data
    current_frame_ids, current_frame_locations = get_frame_data(frame)
    # Debugging: Key variables
    # print(f"Frame {frame}: Current IDs: {current_frame_ids}")

    # Step 3: Determine States (track_states)
    missing_ids, tracked_ids, new_ids = track_states(current_frame_ids, previous_frame_ids)
    # print(f"Frame {frame}: Missing IDs: {missing_ids}")
    # print(f"Frame {frame}: Tracked IDs: {tracked_ids}")
    # print(f"Frame {frame}: New IDs: {new_ids}")

    # Update missing_counts for missing_ids
    for id_ in missing_ids:
        if id_ in missing_counts:
            missing_counts[id_] += 1
        else:
            missing_counts[id_] = 1

    # Step 4: Process missing_ids
    missing_ids_prediction = {}

    for missing_id in missing_ids:
        if missing_id in kalman_filters:
            # Predict the positions for both corners
            kalman_filter_tl = kalman_filters[missing_id]['tl']
            kalman_filter_br = kalman_filters[missing_id]['br']
            kalman_filter_tl.predict()
            kalman_filter_br.predict()
            predicted_tl = kalman_filter_tl.get_current_state()[:2, 0]
            predicted_br = kalman_filter_br.get_current_state()[:2, 0]
            predicted_bbox = [predicted_tl[0], predicted_tl[1], predicted_br[0], predicted_br[1]]
            missing_ids_prediction[missing_id] = predicted_bbox
            predicted_only_ids.append(missing_id)  # Mark as predicted only
            # Update locations_dict with predicted positions
            locations_dict[missing_id] = predicted_bbox
        else:
            # Kalman filter should have been initialized when the object was first detected
            pass

    # Remove predicted objects that are out of frame
    frame_width = 1242  # Update with your frame width
    frame_height = 375  # Update with your frame height
    for missing_id in missing_ids.copy():
        bbox = locations_dict.get(missing_id)
        if bbox and is_bbox_out_of_frame(bbox, frame_width, frame_height):
            # Mark for removal
            missing_ids.remove(missing_id)
            if missing_id in kalman_filters:
                del kalman_filters[missing_id]
            if missing_id in missing_counts:
                del missing_counts[missing_id]
            if missing_id in locations_dict:
                del locations_dict[missing_id]
            about_to_remove_ids.append(missing_id)
            # print(f"Removed Missing ID {missing_id} because it is out of frame")

    # Step 5: Match new_ids to missing_ids_prediction and Initialize Kalman Filters for Unmatched new_ids
    matched_ids, reassociation_map, unmatched_new_ids = match_new_ids_to_missing_predictions(
        new_ids, missing_ids_prediction.copy(), current_frame_locations, max_distance
    )
    # print(f"Frame {frame}: Matched IDs: {matched_ids}")
    # print(f"Frame {frame}: Unmatched New IDs: {unmatched_new_ids}")

    # Initialize Kalman filters for unmatched new_ids
    for new_id in unmatched_new_ids:
        bbox = current_frame_locations[new_id]
        measurement_tl = [bbox[0], bbox[1], depth]
        measurement_br = [bbox[2], bbox[3], depth]
        kalman_filter_tl = KalmanFilter(initial_state=measurement_tl)
        kalman_filter_br = KalmanFilter(initial_state=measurement_br)
        kalman_filters[new_id] = {'tl': kalman_filter_tl, 'br': kalman_filter_br}
        if new_id not in tracked_ids:
            tracked_ids.append(new_id)
        locations_dict[new_id] = bbox
        updated_ids.append(new_id)  # Newly initialized, considered updated

    # Remove matched missing_ids from missing_ids and missing_counts
    for missing_id, new_id in matched_ids:
        if missing_id in missing_ids:
            missing_ids.remove(missing_id)
        if missing_id in missing_counts:
            missing_counts.pop(missing_id)
        # Update Kalman filter mapping if necessary
        kalman_filters[missing_id] = kalman_filters.pop(missing_id, {})
        kalman_filters[missing_id] = kalman_filters.get(missing_id, kalman_filters[new_id])
        # Remove the Kalman filter for new_id if it exists
        if new_id in kalman_filters:
            del kalman_filters[new_id]
        # Update locations_dict
        locations_dict[missing_id] = locations_dict.get(new_id, locations_dict.get(missing_id))
        if new_id in locations_dict:
            del locations_dict[new_id]

    # Step 6: Update Kalman Filter for matched_ids
    for missing_id, new_id in matched_ids:
        # Get Kalman filters
        if missing_id in kalman_filters:
            kalman_filter_tl = kalman_filters[missing_id]['tl']
            kalman_filter_br = kalman_filters[missing_id]['br']
        else:
            # Initialize Kalman filters for missing_id if not present
            bbox = current_frame_locations[new_id]
            measurement_tl = [bbox[0], bbox[1], depth]
            measurement_br = [bbox[2], bbox[3], depth]
            kalman_filter_tl = KalmanFilter(initial_state=measurement_tl)
            kalman_filter_br = KalmanFilter(initial_state=measurement_br)
            kalman_filters[missing_id] = {'tl': kalman_filter_tl, 'br': kalman_filter_br}

        # Predict step
        kalman_filter_tl.predict()
        kalman_filter_br.predict()
        # Update step with new measurements
        bbox = current_frame_locations[new_id]
        measurement_tl = [bbox[0], bbox[1], depth]
        measurement_br = [bbox[2], bbox[3], depth]
        kalman_filter_tl.update(measurement_tl)
        kalman_filter_br.update(measurement_br)
        # Update locations_dict with the updated state
        updated_bbox = extract_bbox_from_state(kalman_filter_tl.get_current_state(), kalman_filter_br.get_current_state())
        locations_dict[missing_id] = updated_bbox
        # Update reassociation_map
        reassociation_map[new_id] = missing_id
        # Replace new_id with missing_id in tracked_ids
        if new_id in tracked_ids:
            tracked_ids.remove(new_id)
        if missing_id not in tracked_ids:
            tracked_ids.append(missing_id)
        updated_ids.append(missing_id)  # Mark as updated with measurement

    # Step 7: Process tracked_ids
    matched_missing_ids = [m[0] for m in matched_ids]
    for tracked_id in tracked_ids:
        if tracked_id in matched_missing_ids:
            continue  # Already updated in Step 6

        # Ensure Kalman filter exists for tracked_id
        if tracked_id not in kalman_filters:
            if tracked_id in current_frame_locations:
                bbox = current_frame_locations[tracked_id]
            else:
                bbox = locations_dict.get(tracked_id, [0, 0, 0, 0])  # Default bbox if not available
            measurement_tl = [bbox[0], bbox[1], depth]
            measurement_br = [bbox[2], bbox[3], depth]
            kalman_filter_tl = KalmanFilter(initial_state=measurement_tl)
            kalman_filter_br = KalmanFilter(initial_state=measurement_br)
            kalman_filters[tracked_id] = {'tl': kalman_filter_tl, 'br': kalman_filter_br}

        kalman_filter_tl = kalman_filters[tracked_id]['tl']
        kalman_filter_br = kalman_filters[tracked_id]['br']
        # Predict step
        kalman_filter_tl.predict()
        kalman_filter_br.predict()
        if tracked_id in current_frame_locations:
            # Update step with measurement
            bbox = current_frame_locations[tracked_id]
            measurement_tl = [bbox[0], bbox[1], depth]
            measurement_br = [bbox[2], bbox[3], depth]
            kalman_filter_tl.update(measurement_tl)
            kalman_filter_br.update(measurement_br)
            updated_ids.append(tracked_id)  # Mark as updated
        else:
            predicted_only_ids.append(tracked_id)  # Mark as predicted only
        # Update locations_dict with the current state
        updated_bbox = extract_bbox_from_state(kalman_filter_tl.get_current_state(), kalman_filter_br.get_current_state())
        locations_dict[tracked_id] = updated_bbox

    # Step 8: Update Tracking States for Next Frame
    # Combine tracked_ids and missing_ids for previous_frame_ids
    previous_frame_ids = tracked_ids.copy()
    for id_ in missing_ids:
        if id_ not in previous_frame_ids:
            previous_frame_ids.append(id_)
    # Update previous_frame_locations for next iteration
    previous_frame_locations = {}
    for id_ in previous_frame_ids:
        if id_ in locations_dict:
            previous_frame_locations[id_] = locations_dict[id_]
        else:
            previous_frame_locations[id_] = None  # Handle missing locations

    # Remove stale missing_ids
    for missing_id in missing_ids.copy():
        if missing_counts[missing_id] > max_missing_frames:
            # Remove object from tracking
            missing_ids.remove(missing_id)
            if missing_id in kalman_filters:
                del kalman_filters[missing_id]
            if missing_id in missing_counts:
                del missing_counts[missing_id]
            if missing_id in locations_dict:
                del locations_dict[missing_id]
            # print(f"Removed Missing ID {missing_id} after exceeding max missing frames")

    # Increment frame_counter
    frame_counter += 1

    # Visualization
    # Read the corresponding image
    image_path = os.path.join(images_folder_path, f"{frame:010d}.png")
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        continue

    # Use Matplotlib for visualization
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Draw bounding boxes
    for track_id, bbox in locations_dict.items():
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox

        # Determine color based on whether the object was updated or predicted only
        if track_id in updated_ids:
            color = 'green'  # Green for updated with measurement
        elif track_id in predicted_only_ids:
            color = 'red'  # Red for predicted only
        elif track_id in about_to_remove_ids:
            color = 'yellow'  # Yellow for about to be removed
        else:
            color = 'cyan'  # Cyan for others (should not occur)

        # Draw the bounding box
        rect = patches.Rectangle((x_top_left, y_top_left), x_bottom_right - x_top_left, y_bottom_right - y_top_left,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Display track_id at the top-left of the bounding box with background
        ax.text(x_top_left, y_top_left - 5, f"ID: {track_id}", color='white', fontsize=8, weight='bold',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))

        # Draw the points used in Kalman filter
        kalman_filter_tl = kalman_filters[track_id]['tl']
        kalman_filter_br = kalman_filters[track_id]['br']

        # Get the predicted position
        predicted_tl = kalman_filter_tl.get_current_state()[:2, 0]
        predicted_br = kalman_filter_br.get_current_state()[:2, 0]

        # Draw predicted positions
        ax.plot(predicted_tl[0], predicted_tl[1], 'o', color='blue', markersize=5)  # Blue for predicted top-left
        ax.plot(predicted_br[0], predicted_br[1], 'o', color='blue', markersize=5)  # Blue for predicted bottom-right

        # If updated with measurement, draw measurement points
        if kalman_filter_tl.updated_with_measurement:
            measurement_tl = kalman_filter_tl.x[:2, 0]
            measurement_br = kalman_filter_br.x[:2, 0]
            ax.plot(measurement_tl[0], measurement_tl[1], 'o', color='yellow', markersize=5)  # Yellow for measurement top-left
            ax.plot(measurement_br[0], measurement_br[1], 'o', color='yellow', markersize=5)  # Yellow for measurement bottom-right

    # Create a legend for the states
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='yellow', lw=2),
                    Line2D([0], [0], color='blue', marker='o', linestyle='None'),
                    Line2D([0], [0], color='yellow', marker='o', linestyle='None')]
    ax.legend(custom_lines, ['Updated', 'Predicted', 'Out of Frame', 'Predicted Point', 'Measurement Point'], loc='upper right')

    plt.axis('on')
    plt.title(f"Frame {frame}")
    plt.tight_layout()
    plt.show()

    # Allow user to close the plot or proceed to the next frame
    # Press 'q' to exit
    user_input = input("Press Enter to continue, or 'q' to quit: ")
    if user_input.lower() == 'q':
        break
