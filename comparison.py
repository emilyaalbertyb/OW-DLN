import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define class ID to class name mapping
class_names = {
    0: "silk_hook",
    1: "yarn_pulling",
    2: "oil_stain",
    3: "lump",
    4: "hole",
    5: "dirt",
    6: "yarn_knot",
}

def draw_chinese_text(img, text, position, font_path, font_size, color):
    """
    Use Pillow to draw Chinese text on OpenCV image
    :param img: OpenCV image
    :param text: Chinese text to draw
    :param position: Top-left position of text (x, y)
    :param font_path: Font file path
    :param font_size: Font size
    :param color: Text color (B, G, R)
    """
    # Convert OpenCV image to Pillow image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # Draw Chinese text
    draw.text(position, text, font=font, fill=(color[0], color[1], color[2], 0))

    # Convert Pillow image back to OpenCV image
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def draw_gt_on_image(image, gt_labels, font_path):
    """
    Draw annotations on image based on GT labels, and display class names (support Chinese).
    """
    img_h, img_w = image.shape[:2]

    for label in gt_labels:
        class_id, x_center, y_center, width, height = label
        # Convert relative coordinates to absolute coordinates
        x_center = x_center * img_w
        y_center = y_center * img_h
        width = width * img_w
        height = height * img_h

        # Calculate top-left and bottom-right coordinates of rectangle
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Get class name, use class_id if not defined
        class_name = class_names.get(class_id, str(class_id))

        # Set color (BGR format) and line thickness
        rectangle_color = (0, 255, 0)  # Green box
        thickness = 20  # Line thickness of 20

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), rectangle_color, thickness)

        # Display class name above box (Chinese)
        text_position = (x_max, y_max+10)  # Adjust text position
        font_size = 100  # Font size
        text_color = (0, 255, 0)  # Red text

        # Use Pillow to draw Chinese text
        image = draw_chinese_text(image, class_name, text_position, font_path, font_size, text_color)

    return image

def load_gt_label(label_file):
    """
    Parse YOLO format GT label file, return label list.
    """
    gt_labels = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            gt_labels.append([int(class_id), x_center, y_center, width, height])
    return gt_labels

def process_and_save_images(pred_folder, gt_label_folder, output_folder, font_path):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through images in prediction folder
    for image_name in os.listdir(pred_folder):
        if image_name.endswith(".bmp") or image_name.endswith(".png"):
            # Read prediction image
            pred_image_path = os.path.join(pred_folder, image_name)
            pred_image = cv2.imread(pred_image_path)

            # Load corresponding GT label based on image name
            gt_label_path = os.path.join(gt_label_folder, image_name.replace('.bmp', '.txt'))  # Assume GT is txt file
            if not os.path.exists(gt_label_path):
                print(f"Label file not found for {image_name}")
                continue
            gt_labels = load_gt_label(gt_label_path)

            # Read GT image and draw annotations
            gt_image = pred_image.copy()
            gt_image = draw_gt_on_image(gt_image, gt_labels, font_path)

            # Concatenate prediction and GT images
            combined_image = np.concatenate((pred_image, gt_image), axis=1)  # Horizontal concatenation

            # Save concatenated image
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, combined_image)
            print(f"Saved combined image: {output_image_path}")

# Set font path
font_path = "C:/Windows/Fonts/simhei.ttf"

pred_folder = 'runs/detect/predict2'
gt_label_folder = 'dataset_liangyou/labels/val'
output_folder = 'comparision'

process_and_save_images(pred_folder, gt_label_folder, output_folder, font_path)
# Call function to process images
