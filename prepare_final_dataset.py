"""
Prepare final dataset for OW-DLN Stage 2 training

Merge known class labels and pseudo-labels for OW-DLN Stage2 training
"""

import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import random


def copy_labels_and_images(src_label_dir, src_image_dir, dst_label_dir, dst_image_dir, 
                           prefix="", class_mapping=None):
    """
    Copy labels and corresponding images
    
    Args:
        src_label_dir: Source label directory
        src_image_dir: Source image directory  
        dst_label_dir: Destination label directory
        dst_image_dir: Destination image directory
        prefix: Filename prefix (to distinguish different sources)
        class_mapping: Class mapping dictionary {old_class: new_class}
        
    Returns:
        int: Number of files copied
    """
    src_label_dir = Path(src_label_dir)
    src_image_dir = Path(src_image_dir)
    dst_label_dir = Path(dst_label_dir)
    dst_image_dir = Path(dst_image_dir)
    
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(src_label_dir.glob('*.txt'))
    count = 0
    
    for label_file in label_files:
        # 查找对应的图像
        img_name = label_file.stem
        img_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = src_image_dir / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"Warning: Cannot find image {img_name}")
            continue
        
        # Read and process labels
        with open(label_file, 'r') as f:
            labels = f.readlines()
        
        # Apply class mapping
        if class_mapping:
            new_labels = []
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    old_cls = int(parts[0])
                    new_cls = class_mapping.get(old_cls, old_cls)
                    parts[0] = str(new_cls)
                    new_labels.append(' '.join(parts))
            labels = new_labels
        else:
            labels = [l.strip() for l in labels]
        
        # Save labels
        new_label_name = f"{prefix}{label_file.name}" if prefix else label_file.name
        new_label_path = dst_label_dir / new_label_name
        
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        # 复制图像
        new_img_name = f"{prefix}{img_path.name}" if prefix else img_path.name
        new_img_path = dst_image_dir / new_img_name
        
        shutil.copy2(img_path, new_img_path)
        count += 1
    
    return count


def merge_datasets(known_label_dir, known_image_dir, 
                   pseudo_label_dir, pseudo_image_dir,
                   output_label_dir, output_image_dir,
                   unknown_class_id=999, split_ratio=0.8, seed=42):
    """
    Merge known class and pseudo-label datasets
    
    Args:
        known_label_dir: Known class label directory
        known_image_dir: Known class image directory
        pseudo_label_dir: Pseudo-label directory
        pseudo_image_dir: Pseudo-label image directory
        output_label_dir: Output label directory (will create train/val subdirectories)
        output_image_dir: Output image directory (will create train/val subdirectories)
        unknown_class_id: Unknown class ID
        split_ratio: Training set ratio
        seed: Random seed
    """
    random.seed(seed)
    
    output_label_dir = Path(output_label_dir)
    output_image_dir = Path(output_image_dir)
    
    # Create train/val directories
    train_label_dir = output_label_dir / 'train'
    val_label_dir = output_label_dir / 'val'
    train_image_dir = output_image_dir / 'train'
    val_image_dir = output_image_dir / 'val'
    
    for d in [train_label_dir, val_label_dir, train_image_dir, val_image_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Preparing Final Dataset")
    print("="*60)
    
    # Step 1: Copy known class data
    print("\n1. Processing known class data...")
    known_count = copy_labels_and_images(
        src_label_dir=known_label_dir,
        src_image_dir=known_image_dir,
        dst_label_dir=train_label_dir,
        dst_image_dir=train_image_dir,
        prefix="known_"
    )
    print(f"   Known class samples: {known_count}")
    
    # Step 2: Process pseudo-label data (containing unknown classes)
    print("\n2. Processing pseudo-label data...")
    pseudo_label_files = list(Path(pseudo_label_dir).glob('*.txt'))
    random.shuffle(pseudo_label_files)
    
    # Split into train and validation sets
    split_idx = int(len(pseudo_label_files) * split_ratio)
    train_pseudo_files = pseudo_label_files[:split_idx]
    val_pseudo_files = pseudo_label_files[split_idx:]
    
    # Copy training set pseudo-labels
    train_pseudo_count = 0
    for label_file in tqdm(train_pseudo_files, desc="   Copying training pseudo-labels"):
        # Find image
        img_name = label_file.stem
        img_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            potential_path = Path(pseudo_image_dir) / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            continue
        
        # Copy label
        dst_label = train_label_dir / f"pseudo_{label_file.name}"
        shutil.copy2(label_file, dst_label)
        
        # Copy image
        dst_image = train_image_dir / f"pseudo_{img_path.name}"
        shutil.copy2(img_path, dst_image)
        
        train_pseudo_count += 1
    
    # Copy validation set pseudo-labels
    val_pseudo_count = 0
    for label_file in tqdm(val_pseudo_files, desc="   Copying validation pseudo-labels"):
        # Find image
        img_name = label_file.stem
        img_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            potential_path = Path(pseudo_image_dir) / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            continue
        
        # Copy label
        dst_label = val_label_dir / f"pseudo_{label_file.name}"
        shutil.copy2(label_file, dst_label)
        
        # Copy image
        dst_image = val_image_dir / f"pseudo_{img_path.name}"
        shutil.copy2(img_path, dst_image)
        
        val_pseudo_count += 1
    
    print(f"   Training pseudo-labels: {train_pseudo_count}")
    print(f"   Validation pseudo-labels: {val_pseudo_count}")
    
    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    # Count training set
    train_labels = list(train_label_dir.glob('*.txt'))
    train_class_counts = {}
    train_total_boxes = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    train_class_counts[cls] = train_class_counts.get(cls, 0) + 1
                    train_total_boxes += 1
    
    # Count validation set
    val_labels = list(val_label_dir.glob('*.txt'))
    val_class_counts = {}
    val_total_boxes = 0
    
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    val_class_counts[cls] = val_class_counts.get(cls, 0) + 1
                    val_total_boxes += 1
    
    print("\nTraining set:")
    print(f"  Images: {len(train_labels)}")
    print(f"  Total boxes: {train_total_boxes}")
    print(f"  Class distribution:")
    for cls in sorted(train_class_counts.keys()):
        cls_name = "unknown" if cls == unknown_class_id else f"class_{cls}"
        print(f"    {cls_name}: {train_class_counts[cls]}")
    
    print("\nValidation set:")
    print(f"  Images: {len(val_labels)}")
    print(f"  Total boxes: {val_total_boxes}")
    print(f"  Class distribution:")
    for cls in sorted(val_class_counts.keys()):
        cls_name = "unknown" if cls == unknown_class_id else f"class_{cls}"
        print(f"    {cls_name}: {val_class_counts[cls]}")
    
    print("\n" + "="*60)
    print("Dataset preparation completed!")
    print(f"Output directory: {output_label_dir.parent}")
    print("="*60)


def create_yaml_config(output_dir, class_names, unknown_class_id=999, yaml_name="data_with_unknown.yaml"):
    """
    Create dataset YAML configuration file
    
    Args:
        output_dir: Output directory
        class_names: List of class names (excluding unknown)
        unknown_class_id: Unknown class ID
        yaml_name: YAML filename
    """
    output_dir = Path(output_dir)
    
    # Ensure unknown class is in the list
    all_classes = class_names.copy()
    if len(all_classes) <= unknown_class_id:
        # Extend list
        all_classes.extend([''] * (unknown_class_id - len(all_classes) + 1))
    all_classes[unknown_class_id] = 'unknown'
    
    yaml_content = f"""# OW-DLN Dataset Configuration
# Contains known classes and unknown classes (pseudo-labels)

path: {output_dir.absolute()}
train: images/train
val: images/val

# Number of classes (including unknown)
nc: {len(all_classes)}

# Class names
names:
"""
    
    for i, name in enumerate(all_classes):
        if name:
            yaml_content += f"  {i}: {name}\n"
    
    yaml_path = output_dir / yaml_name
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\nConfiguration file created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare OW-DLN Stage2 training dataset')
    parser.add_argument('--known_labels', type=str, required=True,
                       help='Known class label directory')
    parser.add_argument('--known_images', type=str, required=True,
                       help='Known class image directory')
    parser.add_argument('--pseudo_labels', type=str, required=True,
                       help='Pseudo-label directory')
    parser.add_argument('--pseudo_images', type=str, required=True,
                       help='Pseudo-label image directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output dataset directory')
    parser.add_argument('--unknown_class_id', type=int, default=999,
                       help='Unknown class ID (default: 999)')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--class_names', type=str, nargs='+',
                       default=['crack', 'scratch', 'dent', 'stain', 'hole'],
                       help='List of known class names')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Merge datasets
    merge_datasets(
        known_label_dir=args.known_labels,
        known_image_dir=args.known_images,
        pseudo_label_dir=args.pseudo_labels,
        pseudo_image_dir=args.pseudo_images,
        output_label_dir=output_dir / 'labels',
        output_image_dir=output_dir / 'images',
        unknown_class_id=args.unknown_class_id,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    
    # Create configuration file
    create_yaml_config(
        output_dir=output_dir,
        class_names=args.class_names,
        unknown_class_id=args.unknown_class_id
    )


if __name__ == '__main__':
    main()

