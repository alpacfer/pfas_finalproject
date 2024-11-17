- `frame_number`: Frame number extracted from the image name
- `track_id`: ID of the tracked object
- `type`: Class name (e.g., 'Car', 'Cyclist', 'Pedestrian')
- `truncated`: Placeholder value (0)
- `occluded`: Placeholder value (0)
- `alpha`: Placeholder value (-10.0)
- `left`: Left coordinate of the bounding box
- `top`: Top coordinate of the bounding box
- `right`: Right coordinate of the bounding box
- `bottom`: Bottom coordinate of the bounding box
- `dimensions`: Placeholder value ("0.0 0.0 0.0")
- `location`: Placeholder value ("0.0 0.0 0.0")
- `rotation_y`: Placeholder value ("-10.0")
- `score`: Confidence score of the detection

## Placeholder Values

The following columns in the KITTI output are filled with placeholder values:
- `truncated`: 0
- `occluded`: 0
- `alpha`: -10.0
- `dimensions`: "0.0 0.0 0.0"
- `location`: "0.0 0.0 0.0"
- `rotation_y`: "-10.0"

## Problems with Occlusion
The track ID for the objects change after occlusion
