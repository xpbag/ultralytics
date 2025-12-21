# https://docs.ultralytics.com/modes/predict/
# from ultralytics import YOLO
# if __name__ == '__main__':
# # Load a pretrained YOLOv8n model
#     model = YOLO("D:\学习日志\目标检测框架源码\YOLO源码\\ultralytics-main\\runs\detect\\train10\weights\\best.pt")
#
# # Define path to the image file
#     source = "C:\\Users\XP\Desktop\\true.jpg"
#
# # Run inference on the source
#     results = model(source)  # list of Results objects

from ultralytics import YOLO
if __name__ == '__main__':
# Load a pretrained YOLOv8n model
    model = YOLO(r"D:\learningJournal\Detection\YOLO\ultralytics-main\xpsUtils\yolov8s_VisDrone_2002\weights\best.pt")
# D:\learningJournal\Detection\YOLO\ultralytics-main\xpsUtils\yolov8s_VisDrone_2002\weights\best.pt
# D:\learningJournal\Detection\YOLO\ultralytics-main\xpsUtils\fasterNet_Neck3CUT_BiFPN_PSA_CPF2D2_2003\weights\best.pt
# Run inference on 'bus.jpg' with arguments
    model.predict(r"D:\Dataset\VisDrone2019-DET\test\images", save=True, imgsz=640, name="YOLOv8s-nolabel-t")