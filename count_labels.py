import os
from collections import defaultdict
from tqdm import tqdm

def count_labels_in_file(label_path):
    """
    Count the number of each class in a single label file
    """
    class_counts = defaultdict(int)
    try:
        with open(label_path, 'r') as f:
            for line in f:
                # YOLO格式：class_id x_center y_center width height
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
    except Exception as e:
        print(f"Error reading {label_path}: {str(e)}")
    return class_counts

def count_all_labels(labels_dir):
    """
    Count class numbers in all label files
    """
    # Store total class counts
    total_counts = defaultdict(int)
    # Store number of images containing each class
    image_counts = defaultdict(int)
    # Record total number of annotation boxes
    total_boxes = 0
    # Record total number of images
    total_images = 0
    
    # Get all txt files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"\nFound {len(label_files)} label files")
    
    # Use tqdm to show processing progress
    for filename in tqdm(label_files, desc="Processing label files"):
        label_path = os.path.join(labels_dir, filename)
        file_counts = count_labels_in_file(label_path)
        
        # Update total counts
        for class_id, count in file_counts.items():
            total_counts[class_id] += count
            image_counts[class_id] += 1
            total_boxes += count
        
        if len(file_counts) > 0:
            total_images += 1
    
    # Print statistics
    print("\n=== Label Statistics ===")
    print(f"Total images: {total_images}")
    print(f"Total annotation boxes: {total_boxes}")
    print("\nClass statistics:")
    print("Class ID | Boxes  | Images with | Avg boxes/img")
    print("---------|--------|-------------|---------------")
    
    # Sort by class ID and output
    for class_id in sorted(total_counts.keys()):
        boxes = total_counts[class_id]
        images = image_counts[class_id]
        avg_per_image = boxes / images if images > 0 else 0
        print(f"{class_id:^9}|{boxes:^8}|{images:^13}|{avg_per_image:^15.2f}")

def main():
    # Set label file directory
    labels_dir = "ZJU-Leaper-YOLO-T1/labels/test"

    # Check if directory exists
    if not os.path.exists(labels_dir):
        print(f"Error: Directory {labels_dir} does not exist!")
        return
    
    # Count labels
    count_all_labels(labels_dir)

if __name__ == "__main__":
    main() 