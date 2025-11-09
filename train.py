from ultralytics import YOLO
import multiprocessing
import wandb

def main():
#
#     # Load a model
#     model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("runs/detect/train2/weights/best.pt")
#     # Train the model
#     # Freeze
#     freeze = [f"model.{0}.", f"model.{1}.", f"model.{2}", f"model.{3}", f"model.{5}", f"model.{6}", f"model.{7}",
#               f"model.{10}", f"model.{11}", f"model.{12}", f"model.{15}", f"model.{16}", f"model.{17}", f"model.{18}",
#               f"model.{19}", f"model.{20}"]  # layers to freeze
#     for k, v in model.named_parameters():
#         v.requires_grad = True  # train all layers
#         if any(x in k for x in freeze):
#             print(f"freezing {k}")
#             v.requires_grad = False
#
#     results = model.train(
#         task="detect",
#         # data="D:/YQM/ultralytics-main/dataset_5+1/dataset.yaml",
#         data="D:/YQM/ultralytics-main/dataset_5+1/dataset.yaml",
#         epochs=1000,
#         batch=8,
#         imgsz=1504,
#         # freeze=4
#     )
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load(
        "D:/ye/ultralytics-main/yolo11s.pt")

    # Train the model
    results = model.train(
        task="detect",
        data="D:/ye/ultralytics-main/ZJU-Leaper-YOLO-T1/data.yaml",
        epochs=1000,
        batch=64,
        imgsz=512,
    )


# def main():
#     # Load a model
#     model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("runs/detect/train2/weights/best.pt")

#     # Define trainable and non-trainable layers
#     # original_conv_layers = [1, 4, 11, 18, 25, 8, 9, 15, 16, 22,23, 28, 29, 30]  # Original conv layers (locked)
#     original_conv_layers = [1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 21, 23, 24, 25]  # Original conv layers (locked)
#     control_conv_layers = [2, 5, 12, 19, 26]  # Control branch conv layers (trainable)
#     zero_conv_layers = [0, 7, 10, 14, 17, 21, 24, 31]  # Zero Conv layers (trainable)

#     # Freeze all layers, then unfreeze layers that need training
#     # for i, (name, param) in enumerate(model.named_parameters()):
#     #     param.requires_grad = False  # Freeze all layers by default
#     #
#     #     # Lock original conv layers
#     #     if any(f"model.{x}." in name for x in original_conv_layers):
#     #         param.requires_grad = False
#     #         print(f"locking {name}")
#     #
#     #     # Train Control branch conv layers and zero conv layers
#     #     if any(f"model.{x}." in name for x in control_conv_layers + zero_conv_layers):
#     #         print(f"training {name}")
#     #         param.requires_grad = True
#     #
#     #         # Initialize zero conv layer weights to 0
#     #         if any(f"model.{x}." in name for x in zero_conv_layers):
#     #             print(f"initializing zero conv {name}")
#     #             param.data.zero_()  # Initialize weights to 0
#     #     if 'model.33.cv2.conv.weight' in name or 'model.33.cv2.conv.bias' in name:
#     #         param.requires_grad = True
#     # # Freeze original conv layers in backbone, but allow detection head to adapt to all classes
#     for i, (name, param) in enumerate(model.named_parameters()):
#         param.requires_grad = True  # Allow all layers to train by default

#         # Only lock original conv layers
#         if any(f"model.{x}." in name for x in original_conv_layers):
#             param.requires_grad = False
#             print(f"locking {name}")
#     # Train the model
#     results = model.train(
#         task="detect",
#         data="dataset_5+1/dataset.yaml",
#         epochs=1000,
#         batch=8,
#         imgsz=1504,
#         warmup_epochs=3,
#         project="yolo_control",
#     )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
