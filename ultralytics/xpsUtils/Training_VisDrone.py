# https://docs.ultralytics.com/modes/train/
# cpu
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
# # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# # model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
#
# # Train the model
# results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
# CLI
# # Build a new model from YAML and start training from scratch
# yolo detect train data=coco8.yaml model=yolov10b.yaml epochs=100 imgsz=640
#
# # Start training from a pretrained *.pt model
# yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
#
# # Build a new model from YAML, transfer pretrained weights to it and start training
# yolo detect train data=coco8.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640

# 多GPU训练
from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
#     model = YOLO("fasterNet_Neck3.yaml").load("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("fasterNet_Neck3CUT_BiFPN_PSA_C2fP.yaml").load(r"D:\learningJournal\Detection\YOLO\ultralytics-main\xpsUtils\runs\detect\yolov8_VisDrone_200\weights\best.pt")  # load a pretrained model (recommended for training)
# # Train the model with 2 GPUs
    results = model.train(data="VisDrone.yaml", epochs=200, imgsz=640, name="fasterNet_Neck3CUT_BiFPN_PSA_C2fP_VisDrone_200", batch=4)

