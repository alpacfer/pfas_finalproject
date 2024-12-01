# Kalman Imrpoved Center

import cv2
import pandas as pd
import os
import numpy as np
import traceback

# Define the file paths

# Sequence 01q
#labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\Sequence Labels\Seq_01'
#labels_file_path = r'C:\Users\User\Documents\dataset\pfas\34759_final_project_rect\seq_01\labels.txt'
#labels_file_path = r'C:\Users\User\Documents\dataset\pfas\34759_final_project_rect\seq_01\labels.txt'
#images_folder_path = r'C:\Users\User\Documents\dataset\pfas\34759_final_project_rect\seq_01\image_02\data'

# Sequence 02
#labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\Sequence Labels\Seq_02.txt'
#labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\kalman\artificial_occlusion_better_02.txt'
labels_file_path = r'C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data\reect\seq_01\labels.txt'
images_folder_path = r'C:\Users\Hasan\OneDrive\Desktop\Projects\pfas_finalproject\data\reect\seq_01\image_02\data'

# Sequence 03
#labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\Sequence Labels\Seq_03.txt'
#labels_file_path = r'C:\Users\User\Documents\GitHub\pfas_finalproject\kalman\filtered_labels_03.txt'
#images_folder_path = r'C:\Users\User\Documents\dataset\pfas\34759_final_project_rect\seq_03\image_02\data'

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
max_distance = 300  # 100 for Sequence 02, 100 for Sequence 01, 300 for Sequence 03
max_missing_frames = 40  # 40 for Sequence 02, 40 for Sequence 01

# Initialize variables
missing_ids = []          # IDs of objects missing in current frame
tracked_ids = []          # IDs of objects being tracked
new_ids = []              # IDs of new detections in current frame
locations_dict = {}       # Stores bounding boxes for each track_id
sizes_dict = {}           # Stores width and height for each track_id
previous_frame_ids = []   # IDs detected in previous frame
previous_frame_locations = {}  # Bounding boxes from previous frame
frame_counter = 1         # Counter for the current frame
reassociation_map = {}    # Maps new_ids to missing_ids
kalman_filters = {}       # Kalman filter instances for each track_id
missing_counts = {}       # Counts how long each missing_id has been missing

# Get sorted list of frames
unique_frames = sorted(df["frame"].unique())

# Kalman Filter Class Definition
class KalmanFilter:
    def __init__(self, initial_state):
        self.state_dim = 6  # x, y, vx, vy, ax, ay
        self.meas_dim = 2  # x, y
        self.dt = dt

        # Initial state vector
        self.x = np.zeros((self.state_dim, 1))
        self.x[0:2, 0] = initial_state  # Initialize position

        # Transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
        ])

        self.base_Q = np.eye(self.state_dim) * 0.1
        self.Q = self.base_Q.copy()
        self.base_R = np.eye(self.meas_dim) * 0.01
        self.R = self.base_R.copy()

        # Initial covariance matrix
        self.P = np.eye(self.state_dim) * 1000

        # Flag to indicate if the filter was updated with a measurement
        self.updated_with_measurement = False

    def predict(self):
        if self.updated_with_measurement:
            # Object was observed; use normal process noise
            self.Q = np.eye(self.state_dim) * 0.1
        else:
            # Object was not observed; increase process noise
            self.Q = np.eye(self.state_dim) * 10.0  # Increase as needed
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.updated_with_measurement = False

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
    

class KalmanFilter3D:
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
        return self.x[:3, 0]  # Return x and y and z positions

def get_frame_data(frame_id):
    frame_data = df[df["frame"] == frame_id]
    frame_ids = frame_data["track id"].unique().tolist()
    frame_locations = {}
    frame_location_3d = {}
    for _, row in frame_data.iterrows():
        track_id = row["track id"]
        bbox = [row["bbox_left"], row["bbox_top"], row["bbox_right"], row["bbox_bottom"]]
        frame_locations[track_id] = bbox
        frame_location_3d[track_id] = [row['location_x'], row['location_y'], row['location_z']]
    return frame_ids, frame_locations, frame_location_3d

def track_states(current_frame_ids, previous_frame_ids):
    missing_ids = [id_ for id_ in previous_frame_ids if id_ not in current_frame_ids]
    tracked_ids = [id_ for id_ in current_frame_ids if id_ in previous_frame_ids]
    new_ids = [id_ for id_ in current_frame_ids if id_ not in previous_frame_ids]
    return missing_ids, tracked_ids, new_ids

def compute_distance(bbox1, bbox2):
    # Compute Euclidean distance between the centers of two bounding boxes
    x1_center, y1_center = calculate_center(bbox1)
    x2_center, y2_center = calculate_center(bbox2)
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def match_new_ids_to_missing_predictions(new_ids, missing_ids_prediction, current_frame_locations, max_distance):
    matched_ids = []
    reassociation_map = {}
    unmatched_new_ids = []

    for new_id in new_ids:
        min_distance = float('inf')
        best_match_id = None
        new_bbox = current_frame_locations[new_id]
        new_center = calculate_center(new_bbox)

        for missing_id, predicted_center in missing_ids_prediction.items():
            distance = np.linalg.norm(new_center - predicted_center)
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

def match_new_ids_to_missing_predictions_3d(new_ids, missing_ids_prediction, current_frame_locations, max_distance):
    matched_ids = []
    reassociation_map = {}
    unmatched_new_ids = []

    for new_id in new_ids:
        min_distance = float('inf')
        best_match_id = None
        new_pose = current_frame_locations[new_id]

        for missing_id, predicted_pose in missing_ids_prediction.items():
            distance = np.linalg.norm(new_pose - predicted_pose)
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

def extract_bbox_from_state(state, size):
    x_center = state[0, 0]
    y_center = state[1, 0]
    width, height = size
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2
    x_bottom_right = x_center + width / 2
    y_bottom_right = y_center + height / 2
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]

def is_bbox_out_of_frame(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    # Check if the bbox is completely outside the frame
    return (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > frame_height)

def calculate_center(bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return np.array([x_center, y_center])

if __name__ == "__main__":

    # Main processing loop
    for frame in unique_frames:
        try:
            print(f"Processing frame {frame}")
            # Clear update tracking lists
            updated_ids = []
            predicted_only_ids = []
            about_to_remove_ids = []

            # Step 2: Retrieve Frame Data
            current_frame_ids, current_frame_locations = get_frame_data(frame)
            print(f"Current frame IDs: {current_frame_ids}")

            # Read the corresponding image
            image_path = os.path.join(images_folder_path, f"{frame:006d}.png")
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            image = cv2.imread(image_path)

            # Verify if the image has been loaded correctly
            if image is None or image.size == 0:
                print(f"Error loading image: {image_path}")
                continue

            frame_height, frame_width = image.shape[:2]

            # Step 3: Determine States (track_states)
            missing_ids, tracked_ids, new_ids = track_states(current_frame_ids, previous_frame_ids)
            print(f"Missing IDs: {missing_ids}")
            print(f"Tracked IDs: {tracked_ids}")
            print(f"New IDs: {new_ids}")

            # Update missing_counts for missing_ids
            for id_ in missing_ids:
                missing_counts[id_] = missing_counts.get(id_, 0) + 1

            # Step 4: Process missing_ids
            missing_ids_prediction = {}
            for missing_id in missing_ids.copy():
                if missing_id in kalman_filters:
                    # Predict the position
                    kalman_filter = kalman_filters[missing_id]
                    kalman_filter.predict()
                    predicted_center = kalman_filter.get_current_state()[:2, 0]
                    # Reconstruct the bbox using the last known size
                    size = sizes_dict.get(missing_id, (0, 0))  # Get width and height
                    predicted_bbox = extract_bbox_from_state(kalman_filter.get_current_state(), size)
                    missing_ids_prediction[missing_id] = predicted_center  # Store predicted center
                    # Update locations_dict
                    locations_dict[missing_id] = predicted_bbox
                    predicted_only_ids.append(missing_id)
                else:
                    print(f"Kalman filter not found for missing ID {missing_id}")
                    missing_ids.remove(missing_id)

            # Remove predicted objects that are out of frame
            for missing_id in missing_ids.copy():
                if missing_id in kalman_filters:
                    kalman_filter = kalman_filters[missing_id]
                    predicted_center = kalman_filter.get_current_state()[:2, 0]
                    
                    # Check if predicted center is out of frame
                    if (predicted_center[0] < 0 or predicted_center[0] > frame_width or 
                        predicted_center[1] < 0 or predicted_center[1] > frame_height):
                        # Remove from tracking
                        print(f"Removed Missing ID {missing_id} because predicted center is out of frame")
                        missing_ids.remove(missing_id)
                        del kalman_filters[missing_id]
                        del missing_counts[missing_id]
                        locations_dict.pop(missing_id, None)
                        sizes_dict.pop(missing_id, None)

            # Step 5: Match new_ids to missing_ids_prediction and Initialize Kalman Filters for Unmatched new_ids
            matched_ids, reassociation_map, unmatched_new_ids = match_new_ids_to_missing_predictions(
                new_ids, missing_ids_prediction.copy(), current_frame_locations, max_distance
            )
            print(f"Matched IDs: {matched_ids}")
            print(f"Unmatched New IDs: {unmatched_new_ids}")

            # Initialize Kalman filters for unmatched new_ids
            for new_id in unmatched_new_ids:
                bbox = current_frame_locations[new_id]
                center = calculate_center(bbox)
                measurement = [center[0], center[1]]
                kalman_filter = KalmanFilter(initial_state=measurement)
                kalman_filters[new_id] = kalman_filter
                sizes_dict[new_id] = (bbox[2] - bbox[0], bbox[3] - bbox[1])  # Store width and height
                if new_id not in tracked_ids:
                    tracked_ids.append(new_id)
                locations_dict[new_id] = bbox
                updated_ids.append(new_id)  # Newly initialized, considered updated

            # Remove matched missing_ids from missing_ids and missing_counts
            for missing_id, new_id in matched_ids:
                if missing_id in missing_ids:
                    missing_ids.remove(missing_id)
                if missing_id in missing_counts:
                    del missing_counts[missing_id]
                # Update Kalman filter mapping if necessary
                if missing_id in kalman_filters:
                    pass
                else:
                    kalman_filters[missing_id] = kalman_filters.pop(new_id)
                # Update sizes_dict
                sizes_dict[missing_id] = sizes_dict.get(missing_id, sizes_dict.get(new_id))
                if new_id in sizes_dict:
                    del sizes_dict[new_id]
                # Update locations_dict
                locations_dict[missing_id] = current_frame_locations[new_id]
                if new_id in locations_dict:
                    del locations_dict[new_id]

            # Step 6: Update Kalman Filter for matched_ids
            for missing_id, new_id in matched_ids:
                # Get Kalman filter for missing_id (ensure it exists)
                kalman_filter = kalman_filters[missing_id]
                # Predict step
                kalman_filter.predict()
                # Update step with new measurements
                bbox = current_frame_locations[new_id]
                center = calculate_center(bbox)
                measurement = [center[0], center[1]]
                kalman_filter.update(measurement)
                # Update sizes_dict with the new size
                sizes_dict[missing_id] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                # Update locations_dict with the updated bbox
                updated_bbox = extract_bbox_from_state(kalman_filter.get_current_state(), sizes_dict[missing_id])
                locations_dict[missing_id] = updated_bbox
                # Replace new_id with missing_id in tracked_ids
                if new_id in tracked_ids:
                    tracked_ids.remove(new_id)
                if missing_id not in tracked_ids:
                    tracked_ids.append(missing_id)
                updated_ids.append(missing_id)  # Mark as updated with measurement

            # Step 7: Process tracked_ids
            matched_missing_ids = [m[0] for m in matched_ids]
            for tracked_id in tracked_ids.copy():
                if tracked_id in matched_missing_ids:
                    continue  # Already updated in Step 6

                # Ensure Kalman filter exists for tracked_id
                if tracked_id not in kalman_filters:
                    if tracked_id in current_frame_locations:
                        bbox = current_frame_locations[tracked_id]
                    else:
                        bbox = locations_dict.get(tracked_id, [0, 0, 0, 0])  # Default bbox if not available
                    center = calculate_center(bbox)
                    measurement = [center[0], center[1]]
                    kalman_filter = KalmanFilter(initial_state=measurement)
                    kalman_filters[tracked_id] = kalman_filter
                    sizes_dict[tracked_id] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    print(f"Initialized Kalman filter for tracked ID {tracked_id}")

                kalman_filter = kalman_filters[tracked_id]
                # Predict step
                kalman_filter.predict()

                if tracked_id in current_frame_locations:
                    # Update step with measurement
                    bbox = current_frame_locations[tracked_id]
                    center = calculate_center(bbox)
                    measurement = [center[0], center[1]]
                    kalman_filter.update(measurement)
                    # Update sizes_dict with the new size
                    sizes_dict[tracked_id] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    # Update locations_dict with the updated bbox
                    updated_bbox = extract_bbox_from_state(kalman_filter.get_current_state(), sizes_dict[tracked_id])
                    locations_dict[tracked_id] = updated_bbox
                    updated_ids.append(tracked_id)  # Mark as updated
                else:
                    # Handle predictions for tracked IDs not in current frame
                    predicted_center = kalman_filter.get_current_state()[:2, 0]
                    # Reconstruct the bbox using the last known size
                    size = sizes_dict.get(tracked_id, (0, 0))
                    predicted_bbox = extract_bbox_from_state(kalman_filter.get_current_state(), size)
                    locations_dict[tracked_id] = predicted_bbox
                    predicted_only_ids.append(tracked_id)

            # Step 8: Update Tracking States for Next Frame
            previous_frame_ids = tracked_ids + missing_ids
            previous_frame_locations = {id_: locations_dict[id_] for id_ in previous_frame_ids if id_ in locations_dict}

            # Remove stale missing_ids
            for missing_id in missing_ids.copy():
                if missing_counts[missing_id] > max_missing_frames:
                    # Remove object from tracking
                    print(f"Removed Missing ID {missing_id} after exceeding max missing frames")
                    missing_ids.remove(missing_id)
                    del kalman_filters[missing_id]
                    del missing_counts[missing_id]
                    locations_dict.pop(missing_id, None)
                    sizes_dict.pop(missing_id, None)

            # Increment frame_counter
            frame_counter += 1

            # Visualization
            # Define a mapping of object types to colors for classification
            classification_colors = {
                "Pedestrian": (34, 139, 34),  # Green
                "Car": (0, 0, 128),         # Navy Blue
                "Cyclist": (139, 0, 0),     # Dark Red
                "misc": (128, 0, 128)       # Purple (for unclassified objects)
            }

            # Draw bounding boxes
            for track_id, bbox in locations_dict.items():
                x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox

                # Get the object type for the current track_id
                # Replace with actual classification lookup logic as needed
                object_type = df[df['track id'] == track_id]['type'].iloc[0] if not df[df['track id'] == track_id].empty else "misc"
                color = classification_colors.get(object_type, (128, 128, 128))  # Default to gray if type is unknown

                # Draw the bounding box with the classification color
                cv2.rectangle(image, (int(x_top_left), int(y_top_left)), (int(x_bottom_right), int(y_bottom_right)), color, 2)

                # Draw text label with background rectangle
                text = f"ID: {track_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size
                # Draw rectangle behind text
                cv2.rectangle(image, (int(x_top_left), int(y_top_left) - text_height - 5),
                            (int(x_top_left) + text_width, int(y_top_left)), color, -1)
                # Draw text
                cv2.putText(image, text, (int(x_top_left), int(y_top_left) - 5), font, font_scale, (255, 255, 255), thickness)

                # Draw the predicted center point
                kalman_filter = kalman_filters[track_id]
                predicted_center = kalman_filter.get_current_state()[:2, 0]

                # Draw predicted center (blue circle)
                cv2.circle(image, (int(predicted_center[0]), int(predicted_center[1])), 5, (255, 0, 0), -1)  # Blue for predicted center

                # If updated with measurement, draw measurement point (yellow circle)
                if kalman_filter.updated_with_measurement:
                    measurement_center = kalman_filter.x[:2, 0]
                    cv2.circle(image, (int(measurement_center[0]), int(measurement_center[1])), 5, (0, 255, 255), -1)  # Yellow for measurement

            # Add frame number to the top-left corner with a background
            frame_text = f"Frame: {frame}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, font_thickness)

            # Set position for text and background
            text_x = 10
            text_y = 30
            bg_x1 = text_x - 5
            bg_y1 = text_y - text_height - 5
            bg_x2 = text_x + text_width + 5
            bg_y2 = text_y + baseline

            # Draw the background rectangle
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

            # Put the text on top of the background
            cv2.putText(image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            # Show the image with the bounding boxes
            cv2.imshow("Image", image)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break

        except Exception as e:
            print(f"Exception occurred at frame {frame}: {e}")
            traceback.print_exc()
            break

    cv2.destroyAllWindows()
