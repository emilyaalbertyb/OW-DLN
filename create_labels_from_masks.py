"""
Create YOLO format detection box labels from DVPS generated masks

This script converts binary masks to bounding box labels for object detection training
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def mask_to_boxes(mask, min_area=100):
    """
    Extract bounding boxes from binary mask
    
    Args:
        mask (np.ndarray): Binary mask image
        min_area (int): Minimum region area threshold
        
    Returns:
        list: Bounding box list [(x1, y1, x2, y2), ...]
    """
    # Ensure it's a binary image
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    
    return boxes


def boxes_to_yolo_format(boxes, img_width, img_height, class_id=0):
    """
    Convert bounding boxes from xyxy format to YOLO format
    
    Args:
        boxes (list): [(x1, y1, x2, y2), ...]
        img_width (int): Image width
        img_height (int): Image height
        class_id (int): Class ID
        
    Returns:
        list: YOLO format labels ["class x_center y_center width height", ...]
    """
    yolo_labels = []
    
    for x1, y1, x2, y2 in boxes:
        # Calculate center point and dimensions (normalized)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Ensure within [0, 1] range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels


def process_masks(masks_dir, output_dir, min_area=100, class_id=0, img_ext='.jpg'):
    """
    Batch process masks and generate YOLO labels
    
    Args:
        masks_dir (Path): Mask directory
        output_dir (Path): Output label directory
        min_area (int): Minimum region area
        class_id (int): Class ID
        img_ext (str): Corresponding image extension
    """
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.jpg'))
    
    print(f"Found {len(mask_files)} mask files")
    
    stats = {
        'total': 0,
        'with_defects': 0,
        'total_boxes': 0
    }
    
    for mask_path in tqdm(mask_files, desc="Processing masks"):
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Cannot read mask {mask_path}")
            continue
        
        img_height, img_width = mask.shape
        
        # Extract bounding boxes
        boxes = mask_to_boxes(mask, min_area=min_area)
        
        stats['total'] += 1
        if len(boxes) > 0:
            stats['with_defects'] += 1
            stats['total_boxes'] += len(boxes)
        
        # Convert to YOLO format
        yolo_labels = boxes_to_yolo_format(boxes, img_width, img_height, class_id=class_id)
        
        # Save label file
        # Remove _mask suffix (if exists)
        label_name = mask_path.stem.replace('_mask', '') + '.txt'
        label_path = output_dir / label_name
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
    
    print("\nProcessing complete!")
    print(f"Total masks: {stats['total']}")
    print(f"Masks with defects: {stats['with_defects']}")
    print(f"Total bounding boxes generated: {stats['total_boxes']}")
    print(f"Average boxes per image: {stats['total_boxes'] / max(stats['with_defects'], 1):.2f}")


def visualize_labels(image_dir, label_dir, mask_dir, output_dir, num_samples=10):
    """
    Visualize generated labels (optional)
    
    Args:
        image_dir (Path): Original image directory
        label_dir (Path): Label directory
        mask_dir (Path): Mask directory
        output_dir (Path): Visualization output directory
        num_samples (int): Number of samples to visualize
    """
    from PIL import Image, ImageDraw
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(label_dir.glob('*.txt'))[:num_samples]
    
    for label_file in label_files:
        # Find corresponding image and mask
        img_name = label_file.stem
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = image_dir / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            continue
        
        # Read image
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Read labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls, x_c, y_c, bw, bh = map(float, parts)
                
                # Convert to pixel coordinates
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # Save
        output_path = output_dir / f"{img_name}_labeled.jpg"
        img.save(output_path)
    
    print(f"\nVisualization complete! Results saved at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO format labels from masks')
    parser.add_argument('--masks_dir', type=str, required=True,
                       help='Input mask directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output label directory')
    parser.add_argument('--min_area', type=int, default=100,
                       help='Minimum region area threshold (default: 100)')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID (default: 0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization results')
    parser.add_argument('--image_dir', type=str,
                       help='Original image directory (for visualization)')
    parser.add_argument('--vis_output', type=str, default='visualizations',
                       help='Visualization output directory')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Process masks
    process_masks(
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        class_id=args.class_id
    )
    
    # Visualize (if needed)
    if args.visualize and args.image_dir:
        visualize_labels(
            image_dir=args.image_dir,
            label_dir=args.output_dir,
            mask_dir=args.masks_dir,
            output_dir=args.vis_output,
            num_samples=args.num_vis
        )


if __name__ == '__main__':
    main()
