from attr import dataclass
from numpy.lib.npyio import savetxt

from ultralytics import YOLO
if __name__ == "__main__":
# Load a pretrained YOLO11n model
    model = YOLO("D:/YQM/ultralytics-main/runs/detect/train16/weights/best.pt")

    # Define path to directory containing images and videos for inference
    source = "D:/YQM/ultralytics-main/dataset_5+1/images/train"
    # New class name mapping
    # Add at line 338 in ultralytics/engine/predictor.py
# new_names = {
#             0: 'hole',  # 0
#             #  1: sansi  #1
#             1: 'mispick',
#             2: 'float',  # 2
#             3: 'hardSize',  #
#             4: 'warpingyarn',  #
#             5: 'unknow'
#         }
#         result.names = new_names
    # Run inference on the source
    results = model(source, save_txt=True,save=True,imgsz=1024)  # generator of Results objects

    # # Old class names (during training)
    # old_names = model.names
    #
    # # New class name mapping
    # new_names = {
    #     0: 'hole' , # 0
    #     #  1: sansi  #1
    #     1: 'mispick',
    #     2: 'float'  ,# 2
    #     3: 'hardSize',  #
    #     4: 'warpingyarn',  #
    #     5: 'unknow'
    # }
    #
    # # Parse detection results and replace class names
    # for result in results:
    #     for box in result.boxes:
    #         cls_id = int(box.cls)  # Get class index
    #         old_name = old_names[cls_id]  # Original class name
    #         new_name = new_names.get(cls_id, old_name)  # Replace class name (default unchanged)
    #         print(f"Class index: {cls_id}, Old class name: {old_name}, New class name: {new_name}")