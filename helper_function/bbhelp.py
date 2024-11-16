import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageOps
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import plotly.graph_objs as go
from plotly.offline import plot
import random


def extract_image_info(image_index, labels_file):
    """
    Extracts and returns all relevant information for a given image index from the labels file,
    organized in a list containing information for all objects in the frame.

    Parameters:
    image_index (int): The index of the image to extract information for.
    labels_file (str): Path to the labels.txt file containing bounding box information.

    Returns:
    list: A list of dictionaries, each containing information about an object in the image.
    """
    # Load labels from the file
    with open(labels_file, 'r') as file:
        labels = file.readlines()

    # Filter labels for the current frame
    frame_number = image_index
    frame_info = []

    for line in labels:
        columns = line.strip().split()
        if int(columns[0]) == frame_number:
            track_id_raw = columns[1]
            try:
                track_id = int(track_id_raw)
            except ValueError:
                track_id = track_id_raw  # Keep it as string if it cannot be converted

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
            # Append info to the list
            frame_info.append(info)

    return frame_info

def get_track_info(track_id, image_index, labels_file, item_key):
    """
    Retrieves the specified item for a given track_id in a specific frame.
    
    Parameters:
    track_id (int): The track ID of the object to retrieve.
    image_index (int): The frame number (image index) to search in.
    labels_file (str): Path to the labels.txt file containing bounding box information.
    item_key (str): The key of the item to fetch (e.g., 'center', 'bbox', 'type').
    
    Returns:
    The value corresponding to the item_key for the matching object.
    If no matching object is found, returns None.
    """
    # Extract frame information for the given frame
    frame_info = extract_image_info(image_index, labels_file)
    
    # Iterate over objects in the frame to find the one with the desired track_id
    for obj in frame_info:
        if obj['track_id'] == track_id:
            return obj.get(item_key)
    
    # If no object is found with the given track_id, return None
    return None


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
    image_files = [entry.name for entry in os.scandir(images_folder)
                   if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
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
    for obj in frame_info:
        left, top, right, bottom = obj['bbox']
        obj_type = obj['type']
        track_id = obj['track_id']
        track_id_text = f"ID: {track_id}"
        occluded = obj['occluded']
        color = type_colors.get(obj_type, ('yellow', 'darkyellow'))[1 if occluded else 0]

        if display_type == 'bbox':
            # Draw bounding box
            rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # Display track_id at the top-left of the bounding box with background
            ax.text(left, top - 5, track_id_text, color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))
        elif display_type == 'center':
            # Plot center point
            center_x, center_y = obj['center']
            ax.plot(center_x, center_y, 'o', color=color, markersize=5)
            # Display track_id near the center point with background
            ax.text(center_x + 5, center_y - 5, track_id_text, color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))

    # Create a legend for the types of objects
    handles = [patches.Patch(color=colors[0], label=obj_type) for obj_type, colors in type_colors.items()]
    ax.legend(handles=handles, loc='upper right')

    plt.axis('on')
    plt.title(f"Frame {image_index}: {image_files[image_index]}")
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

    image_files = [entry.name for entry in os.scandir(images_folder)
                   if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    image_files.sort()  # Ensure consistent ordering

    for image_index, image_name in enumerate(image_files):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        # Extract frame information for the current image index
        frame_info = extract_image_info(image_index, labels_file)

        # Display frame number at the top-left corner
        cv2.putText(image, f'Frame ID: {image_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for obj in frame_info:
            left, top, right, bottom = map(int, obj['bbox'])
            obj_type = obj['type']
            track_id = obj['track_id']
            track_id_text = f"ID: {track_id}"
            color = type_colors.get(obj_type, (0, 255, 255))

            if display_type == 'bbox':
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                label = track_id_text
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (left, top - text_height - 10), (left + text_width, top), color, -1)
                cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            elif display_type == 'center':
                center_x, center_y = map(int, obj['center'])
                cv2.circle(image, (center_x, center_y), 5, color, -1)
                label = track_id_text
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (center_x + 10, center_y - text_height - 5),
                              (center_x + 10 + text_width, center_y + 5), color, -1)
                cv2.putText(image, label, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Video Playback", image)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def create_obstructed_labels_complex(labels_file, obstruction_data):
    """
    Creates a new labels file with specified track IDs removed within specific frame ranges,
    simulating obstruction for each track ID in its respective range.

    Parameters:
    labels_file (str): Path to the original labels file.
    obstruction_data (dict): Dictionary where each key is a track_id and each value is a tuple (start_frame, end_frame).

    Returns:
    None
    """
    # Read original labels
    with open(labels_file, 'r') as file:
        labels = file.readlines()

    # Prompt for the name of the new labels file
    new_labels_file = input("Enter the name for the new labels file: ")

    # Prepare modified data
    modified_labels = []
    for line in labels:
        columns = line.strip().split()
        frame = int(columns[0])
        current_track_id = int(columns[1])

        # Check if the current track_id has an obstruction range
        if current_track_id in obstruction_data:
            start_frame, end_frame = obstruction_data[current_track_id]
            # Skip lines within the obstruction range for this track_id
            if start_frame <= frame <= end_frame:
                continue  # Skip this line

        # Add line to modified labels if it doesn't match any obstruction criteria
        modified_labels.append(line)

    # Write modified labels to new file
    with open(new_labels_file, 'w') as file:
        file.writelines(modified_labels)

    print(f"New labels file created: {new_labels_file}")

def anonymize_track_ids(labels_file):
    """
    Creates a new labels file with all track IDs replaced by '??', retaining other information.

    Parameters:
    labels_file (str): Path to the original labels file.

    Returns:
    None
    """
    # Read original labels
    with open(labels_file, 'r') as file:
        labels = file.readlines()

    # Prompt for the name of the new labels file
    new_labels_file = input("Enter the name for the new labels file with anonymized track IDs: ")

    # Prepare modified data with "??" replacing the track_id column (second column)
    modified_labels = []
    for line in labels:
        columns = line.strip().split()
        # Replace the track_id with '??'
        columns[1] = "??"
        # Reconstruct the line with the modified track_id and add to the list
        modified_line = " ".join(columns) + "\n"
        modified_labels.append(modified_line)

    # Write modified labels to the new file
    with open(new_labels_file, 'w') as file:
        file.writelines(modified_labels)

    print(f"New labels file with anonymized track IDs created: {new_labels_file}")


def plot_trajectories_on_initial_frame(labels_file, images_folder, track_ids=None):
    """
    Plots the center trajectories of bounding boxes for specified track IDs
    overlaid on a darkened initial frame of the sequence.
    
    Parameters:
    labels_file (str): Path to the labels.txt file containing bounding box information.
    images_folder (str): Path to the folder containing the sequence images.
    track_ids (list): List of track IDs to plot. If None, plot all track IDs.
    
    Returns:
    None
    """
    # Read labels file and extract data
    with open(labels_file, 'r') as file:
        labels = file.readlines()
    
    # Dictionary to store center coordinates for each track ID
    track_data = {}
    
    for line in labels:
        columns = line.strip().split()
        frame = int(columns[0])
        track_id_raw = columns[1]
        try:
            track_id = int(track_id_raw)
        except ValueError:
            track_id = track_id_raw  # Keep as string if not convertible to int
        
        left, top, right, bottom = map(float, columns[6:10])
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        # Append center coordinates to the corresponding track ID
        if track_id not in track_data:
            track_data[track_id] = []
        track_data[track_id].append((frame, center_x, center_y))
    
    # Filter track IDs to plot if a specific list is provided
    if track_ids is not None:
        track_data = {tid: coords for tid, coords in track_data.items() if tid in track_ids}
    
    # Load and process the first image as the background
    image_files = [entry.name for entry in os.scandir(images_folder)
                   if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    image_files.sort()
    first_image_path = os.path.join(images_folder, image_files[0])
    image = Image.open(first_image_path).convert('L')  # Convert to grayscale
    image = ImageEnhance.Brightness(image).enhance(0.8)  # Dim the brightness
    image = ImageOps.colorize(image, black="black", white="darkgray")  # Apply a dark tone

    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title("Track ID Trajectories on Darkened Initial Frame", fontsize=16, color="black")
    plt.axis('off')  # Hide axes since we're overlaying on the image
    
    # Generate a unique, bright color for each track ID
    colors = {}
    for track_id in track_data.keys():
        colors[track_id] = [random.random(), random.random(), random.random()]
    
    # Overlay trajectories for each track ID
    for track_id, trajectory in track_data.items():
        x_coords = [coord[1] for coord in trajectory]
        y_coords = [coord[2] for coord in trajectory]
        plt.plot(x_coords, y_coords, label=f'Track ID {track_id}', color=colors[track_id], linewidth=2)
    
    # Add a legend with adjusted text color and visibility
    legend = plt.legend(
        loc='upper left',
        fontsize=8,
        bbox_to_anchor=(1, 1),
        frameon=True,
        facecolor="black",
        edgecolor="white"
    )
    for text in legend.get_texts():
        text.set_color("white")  # Set legend text color to white
    
    # Adjust layout to ensure everything fits without extra margins
    plt.tight_layout()
    plt.show()


def plot_trajectories_in_3d_interactive(labels_file, track_ids=None):
    """
    Plots interactive 3D trajectories of specified track IDs using the 'location' data from the labels file,
    with the beginnings and ends clearly marked. Y is treated as the vertical axis.
    
    Parameters:
    labels_file (str): Path to the labels.txt file containing bounding box and location information.
    track_ids (list): List of track IDs to plot. If None, plot all track IDs.
    
    Returns:
    None
    """
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
    import random

    # Read labels file and extract data
    data = []
    with open(labels_file, 'r') as file:
        for line in file:
            columns = line.strip().split()
            frame = int(columns[0])
            track_id_raw = columns[1]
            try:
                track_id = int(track_id_raw)
            except ValueError:
                track_id = track_id_raw  # Keep as string if not convertible to int
            
            # Extract the 'location' data (x, y, z)
            x = float(columns[13])  # x coordinate
            y = float(columns[14])  # y coordinate (vertical axis)
            z = float(columns[15])  # z coordinate
            
            # Append data to list
            data.append({
                'Frame': frame,
                'TrackID': track_id,
                'X': x,
                'Y': y,
                'Z': z
            })

    # Create a DataFrame for easier handling
    df = pd.DataFrame(data)

    # Filter track IDs if specified
    if track_ids is not None:
        df = df[df['TrackID'].isin(track_ids)]

    # Get unique track IDs
    unique_track_ids = df['TrackID'].unique()

    # Define a color palette
    color_map = px.colors.qualitative.Light24
    color_dict = {track_id: color_map[i % len(color_map)] for i, track_id in enumerate(unique_track_ids)}

    # Prepare the data for Plotly
    data_traces = []
    for track_id in unique_track_ids:
        track_df = df[df['TrackID'] == track_id].sort_values('Frame')
        x_coords = track_df['X']
        y_coords = track_df['Y']
        z_coords = track_df['Z']
        frames = track_df['Frame']

        color = color_dict[track_id]

        # Create the line trace for the trajectory
        trace_line = go.Scatter3d(
            x=x_coords,
            y=z_coords,  # z is plotted on the horizontal plane
            z=y_coords,  # y is the vertical axis
            mode='lines',
            name=f'Track ID {track_id}',
            line=dict(color=color, width=4),
            hoverinfo='text',
            text=[f'Track ID: {track_id}<br>Frame: {frame}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
                  for frame, x, y, z in zip(frames, x_coords, y_coords, z_coords)]
        )

        # Mark the starting point
        trace_start = go.Scatter3d(
            x=[x_coords.iloc[0]],
            y=[z_coords.iloc[0]],
            z=[y_coords.iloc[0]],
            mode='markers+text',
            marker=dict(color='green', size=8, symbol='circle'),
            name=f'Start of {track_id}',
            hoverinfo='text',
            text=[f'Start<br>Track ID: {track_id}<br>Frame: {frames.iloc[0]}'],
            textposition='top center',
            showlegend=False
        )

        # Mark the ending point
        trace_end = go.Scatter3d(
            x=[x_coords.iloc[-1]],
            y=[z_coords.iloc[-1]],
            z=[y_coords.iloc[-1]],
            mode='markers+text',
            marker=dict(color='red', size=8, symbol='x'),
            name=f'End of {track_id}',
            hoverinfo='text',
            text=[f'End<br>Track ID: {track_id}<br>Frame: {frames.iloc[-1]}'],
            textposition='top center',
            showlegend=False
        )

        # Add the traces to the data list
        data_traces.extend([trace_line, trace_start, trace_end])

    # Define the layout of the plot
    layout = go.Layout(
        title='Interactive 3D Trajectories of Track IDs',
        scene=dict(
            xaxis=dict(title='X (meters)'),
            yaxis=dict(title='Z (meters)'),  # Z is horizontal plane axis
            zaxis=dict(title='Y (meters)'),  # Y is vertical axis
            aspectmode='data'  # Ensures equal scaling for all axes
        ),
        legend=dict(
            x=0.8,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Create the figure and display it
    fig = go.Figure(data=data_traces, layout=layout)
    fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)))
    fig.show()

