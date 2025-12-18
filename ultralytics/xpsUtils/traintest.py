
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11_cut.yaml").load(r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="coco8.yaml", epochs=1, imgsz=640, name="test", batch=8)

