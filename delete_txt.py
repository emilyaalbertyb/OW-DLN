# Compare image names in folder with annotation paths, delete excess txt files
import os
import cv2
import numpy as np

def delete_txt(image_folder, txt_folder):
    # Get image names in folder
    image_names = os.listdir(image_folder)
    image_names = [os.path.splitext(name)[0] for name in image_names]
    # Get txt file names in folder
    txt_names = os.listdir(txt_folder)
    txt_names = [os.path.splitext(name)[0] for name in txt_names]

    # Iterate through txt files
    for txt_name in txt_names:
        # Get txt file name without extension
        txt_name = os.path.splitext(txt_name)[0]
        # If txt file name is not in image names, delete txt file
        if txt_name not in image_names:
            os.remove(os.path.join(txt_folder, txt_name+'.txt'))
            print(f"Deleted {txt_name}.txt")
    for img_name in image_names:
        # Get txt file name without extension
        # txt_name = os.path.splitext(txt_name)[0]
        # If txt file name is not in image names, delete txt file
        if img_name not in txt_names:
            os.remove(os.path.join(image_folder, img_name+'.png'))
            print(f"Deleted {img_name}.jpg")

if __name__ == '__main__':
    image_folder = 'D:/ye/ultralytics-main/dataset1/images/val'  # Image folder path
    txt_folder = 'D:/ye/ultralytics-main/dataset1/labels/val'  # txt folder path
    delete_txt(image_folder, txt_folder)

