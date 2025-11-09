import os

def change_image_extensions_to_png(folder_path):
    """
    将指定文件夹下的所有图片文件后缀改为.png
    :param folder_path: 目标文件夹路径
    """
    # 支持的常见图片格式
    image_extensions = ['.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.jfif']
    
    for filename in os.listdir(folder_path):
        # 获取文件的基本名和扩展名
        base, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        
        # 如果是图片文件且不是png
        if ext_lower in image_extensions:
            # 构造新文件名
            new_filename = base + '.png'
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f'已重命名: {filename} -> {new_filename}')
            except Exception as e:
                print(f'重命名 {filename} 失败: {e}')

# 使用示例
if __name__ == '__main__':
    folder_path = input('请输入文件夹路径: ')
    if os.path.isdir(folder_path):
        change_image_extensions_to_png(folder_path)
    else:
        print('错误: 指定的路径不是一个有效的文件夹')