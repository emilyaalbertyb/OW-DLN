import json
import os


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box from [x_min, y_min, x_max, y_max] to YOLO format.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def convert_json_to_yolo(json_file, output_dir, img_width, img_height):
    """
    Convert a JSON annotation file to YOLO format with fixed image dimensions.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    # Group annotations by image name
    grouped_annotations = {}
    for ann in annotations:
        name = ann['name']
        defect_name = ann['defect_name']
        bbox = ann['bbox']
        if name not in grouped_annotations:
            grouped_annotations[name] = []
        grouped_annotations[name].append((defect_name, bbox))

    # Process each image's annotations
    for name, anns in grouped_annotations.items():
        yolo_annotations = []
        for defect_name, bbox in anns:
            x_center, y_center, width, height = convert_bbox_to_yolo_format(bbox, img_width, img_height)
            yolo_annotations.append(f"{defect_name} {x_center} {y_center} {width} {height}")

        # Save YOLO annotations to a txt file
        txt_file_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.txt")
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write("\n".join(yolo_annotations))


# Example usage
json_file = 'defect_dataset/anno_train.json'  # Path to your JSON file
output_dir = 'yolo_annotations'  # Directory to save YOLO txt files
img_width, img_height = 2446, 1000  # Fixed image dimensions

convert_json_to_yolo(json_file, output_dir, img_width, img_height)
