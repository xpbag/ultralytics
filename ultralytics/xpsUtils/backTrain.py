from ultralytics import YOLO

# 关键：加载上次训练的last.pt文件（而非初始模型）
if __name__ == '__main__':
    model = YOLO(r"D:\learningJournal\Detection\ultralytics\runs\detect\train5\weights\last.pt")

    # 继续训练（注意：epochs需设置为最终目标轮数，而非剩余轮数）
    # 例如：首次训练到50轮中断，目标是300轮，仍设置epochs=300，YOLO会从50轮继续到300轮
    results = model.train(
        data="VisDrone.yaml",
        epochs=300,  # 最终目标轮数
        batch=4,
        imgsz=640,
        patience=50,
        project=r"D:\learningJournal\Detection\ultralytics\runs\detect\train5",
        name="train5",  # 保持与首次训练相同的实验名，新的训练记录会追加到原目录
        resume=True  # 显式指定恢复训练（可选，YOLO加载last.pt时会自动启用）
    )