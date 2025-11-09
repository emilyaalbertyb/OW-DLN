# import os
# from PIL import Image
#
#
# def resize_images_in_folder(folder_path, target_size=(512, 512)):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#
#         # 只处理图片文件
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             try:
#                 # 打开图片
#                 with Image.open(file_path) as img:
#                     # 调整图片大小
#                     resized_img = img.resize(target_size, resample=Image.BICUBIC)
#                     # 保存回原文件
#                     resized_img.save(file_path)
#                     print(f"已调整大小并保存: {filename}")
#             except Exception as e:
#                 print(f"处理图片 {filename} 时发生错误: {e}")
#
#
# if __name__ == "__main__":
#     # 设置文件夹路径
#     folder_path = "data/inpainting_fabric/normal_img"  # 请替换为你的文件夹路径
#     resize_images_in_folder(folder_path)

import os

def rename_images(folder_path, suffix_to_remove):
    """
    Rename images in the given folder by removing the specified suffix from their filenames.

    Args:
        folder_path (str): Path to the folder containing the images.
        suffix_to_remove (str): The suffix to remove from the filenames.
    """
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        # Check if the file is an image and contains the suffix
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and suffix_to_remove in filename:
            old_path = os.path.join(folder_path, filename)
            # Remove the suffix from the filename
            new_filename = filename.replace(suffix_to_remove, "")
            new_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Replace with your folder path and suffix to remove
folder_path = "data/inpainting_fabric/output_inpainting_original_paper"  # 替换为你的文件夹路径
suffix_to_remove = "_sd_examples_NOT_EMA_last.ckpt"  # 替换为要移除的后缀

rename_images(folder_path, suffix_to_remove)
