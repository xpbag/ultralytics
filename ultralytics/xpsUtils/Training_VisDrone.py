
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11s.yaml").load("D:\learningJournal\Detection\\ultralytics\\ultralytics\\xpsUtils\models\\best.pt")

# Train the model
    results = model.train(data="D:\learningJournal\Detection\\ultralytics\\ultralytics\\xpsUtils\dataSet\VisDrone.yaml", epochs=300, imgsz=640, batch=8)
