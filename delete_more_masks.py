#分别对比train和val下的masks和defect_images，删除多余的mask，去掉_mask.png后缀进行比较
import os
import shutil

def delete_extra_masks(dataset_root):
    for split in ['train', 'val']:
        masks_dir = os.path.join(dataset_root, split, 'masks')
        defect_dir = os.path.join(dataset_root, split, 'defect_images')
        if os.path.exists(masks_dir) and os.path.exists(defect_dir):
            masks = [f for f in os.listdir(masks_dir) if f.endswith(('_mask.png'))]
            defect_images = [f for f in os.listdir(defect_dir) if f.endswith(('.jpg', '.png'))]
            for mask in masks:
                if mask.replace('_mask.png', '.jpg') not in defect_images:
                    os.remove(os.path.join(masks_dir, mask))
                    print(f"Deleted mask: {mask}")
        else:
            print(f"Masks directory does not exist: {masks_dir}")
            print(f"Defect images directory does not exist: {defect_dir}")

if __name__ == "__main__":
    dataset_root = 'unet/dataset'
    delete_extra_masks(dataset_root)