import torch

from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = (YOLO(r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\cfg\models\11xps\yolo11_LSNet.yaml")
             .load(r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\yolo11n.pt"))
    # load a pretrained model (recommended for training)
    results = model.train(data=r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\cfg\dataSet\VisDrone-puls.yaml", epochs=1, imgsz=640, name="test", batch=8, device="0")

