import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def extract_image_info(image_index, labels_file):
    """
    Extracts and returns all relevant information for a given image index from the labels file,
    organized in a dictionary keyed by track_id for easy access.

    Parameters:
    image_index (int): The index of the image to extract information for.
    labels_file (str): Path to the labels.txt file containing bounding box information.

    Returns:
    dict: A dictionary where each key is a track_id, and each value is a dictionary containing
          information about an object in the image.
    """
    # Load labels from the file
    with open(labels_file, 'r') as file:
        labels = file.readlines()

    # Filter labels for the current frame
    frame_number = image_index
    frame_info = {}

    for line in labels:
        columns = line.strip().split()
        if int(columns[0]) == frame_number:
            track_id = int(columns[1])
            left, top, right, bottom = map(float, columns[6:10])
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2

            info = {
                'frame': int(columns[0]),
                'track_id': track_id,
                'type': columns[2],
                'truncated': float(columns[3]),
                'occluded': int(columns[4]),
                'alpha': float(columns[5]),
                'bbox': [left, top, right, bottom],
                'center': [center_x, center_y],
                'dimensions': [float(columns[10]), float(columns[11]), float(columns[12])],
                'location': [float(columns[13]), float(columns[14]), float(columns[15])],
                'rotation_y': float(columns[16]),
                'score': float(columns[17]) if len(columns) > 17 else None
            }
            # Use track_id as the key for easy access
            frame_info[track_id] = info

    return frame_info


def display_image(image_index, images_folder, labels_file, display_type='bbox'):
    """
    Display an image given its index from a specified folder, with either bounding boxes or centers, and track IDs.

    Parameters:
    image_index (int): The index of the image to display.
    images_folder (str): Path to the folder containing the images.
    labels_file (str): Path to the labels.txt file containing bounding box information.
    display_type (str): 'bbox' to display bounding boxes, 'center' to display only the center.
    """
    type_colors = {'Car': ('red', 'darkred'), 'Pedestrian': ('blue', 'darkblue'), 'Cyclist': ('green', 'darkgreen')}

    # Get image files
    image_files = [entry.name for entry in os.scandir(images_folder) if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    image_files.sort()

    if image_index < 0 or image_index >= len(image_files):
        print(f"Invalid image index. Please choose a number between 0 and {len(image_files) - 1}.")
        return

    image_path = os.path.join(images_folder, image_files[image_index])
    frame_info = extract_image_info(image_index, labels_file)

    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Iterate through all objects in the current frame
    for obj in frame_info.values():  # Access values directly from the dictionary
        left, top, right, bottom = obj['bbox']
        obj_type = obj['type']
        track_id = obj['track_id']
        occluded = obj['occluded']
        color = type_colors.get(obj_type, ('yellow', 'darkyellow'))[1 if occluded else 0]

        if display_type == 'bbox':
            # Draw bounding box
            rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # Display track_id at the top-left of the bounding box with background
            ax.text(left, top - 5, f'ID: {track_id}', color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))
        elif display_type == 'center':
            # Plot center point
            center_x, center_y = obj['center']
            ax.plot(center_x, center_y, 'o', color=color, markersize=5)
            # Display track_id near the center point with background
            ax.text(center_x + 5, center_y - 5, f'ID: {track_id}', color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))

    # Create a legend for the types of objects
    handles = [patches.Patch(color=colors[0], label=obj_type) for obj_type, colors in type_colors.items()]
    ax.legend(handles=handles, loc='upper right')

    plt.axis('on')
    plt.title(image_path)
    plt.show()


def display_images_as_video(images_folder, labels_file, frame_delay=500, display_type='bbox'):
    """
    Display all images in a folder like a video, showing bounding boxes or centers with track IDs.

    Parameters:
    images_folder (str): Path to the folder containing the images.
    labels_file (str): Path to the labels.txt file containing bounding box information.
    frame_delay (int): Delay between frames in milliseconds (default is 500 ms).
    display_type (str): 'bbox' to display bounding boxes, 'center' to display only the center.
    """
    type_colors = {'Car': (0, 0, 255), 'Pedestrian': (255, 0, 0), 'Cyclist': (0, 255, 0)}

    image_files = [entry.name for entry in os.scandir(images_folder) if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    image_files.sort()  # Ensure consistent ordering

    for image_index, image_name in enumerate(image_files):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        # Extract frame information for the current image index
        frame_info = extract_image_info(image_index, labels_file)

        # Display frame number at the top-left corner
        cv2.putText(image, f'Frame ID: {image_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for obj in frame_info.values():
            left, top, right, bottom = map(int, obj['bbox'])
            obj_type = obj['type']
            track_id = obj['track_id']
            color = type_colors.get(obj_type, (0, 255, 255))

            if display_type == 'bbox':
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                label = f'ID: {track_id}'
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (left, top - text_height - 10), (left + text_width, top), color, -1)
                cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            elif display_type == 'center':
                center_x, center_y = map(int, obj['center'])
                cv2.circle(image, (center_x, center_y), 5, color, -1)
                label = f'ID: {track_id}'
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (center_x + 10, center_y - text_height - 5), 
                              (center_x + 10 + text_width, center_y + 5), color, -1)
                cv2.putText(image, label, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Video Playback", image)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()