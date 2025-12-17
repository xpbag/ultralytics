from ultralytics import YOLO
import logging
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------------- 日志配置函数 --------------------------
def setup_logger(log_dir="./yolo_train_logs"):
    """
    配置日志器，同时输出到控制台和文件
    :param log_dir: 日志保存目录
    :return: 配置好的logger对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    # 日志文件命名：时间戳+train.log
    log_file = os.path.join(log_dir, f"yolo11s_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 创建日志器
    logger = logging.getLogger("YOLO_Train_Logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 清除原有处理器，避免重复输出

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# -------------------------- 邮件提醒函数 --------------------------
def send_train_notification(
        subject, content,
        sender_email="xyyniao@163.com",
        sender_password="RVT2g337LNH4chD8",
        receiver_email="xyyniao@qq.com",
        smtp_server="smtp.163.com",
        smtp_port=465
):
    """
    发送训练结束/异常提醒邮件
    :param subject: 邮件主题
    :param content: 邮件内容
    :param sender_email: 发件人邮箱
    :param sender_password: 邮箱授权码（非登录密码）
    :param receiver_email: 收件人邮箱（可多个，用逗号分隔）
    :param smtp_server: 邮箱SMTP服务器（如163: smtp.163.com，QQ: smtp.qq.com）
    :param smtp_port: SMTP端口（SSL通常为465）
    """
    # 构建邮件
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # 添加邮件正文
    msg.attach(MIMEText(content, "plain", "utf-8"))

    try:
        # 连接SMTP服务器并发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email.split(","), msg.as_string())
        logger.info("训练提醒邮件发送成功！")
    except Exception as e:
        logger.error(f"邮件发送失败：{str(e)}", exc_info=True)

# -------------------------- 主训练流程 --------------------------
if __name__ == '__main__':
    # 初始化日志
    logger = setup_logger()

    # 训练配置信息
    model_yaml = "yolo11n.yaml"
    pretrained_weights = "D:\learningJournal\Detection\\ultralytics\\ultralytics\\xpsUtils\yolo11n.pt"
    data_yaml = "D:\learningJournal\Detection\\ultralytics\\ultralytics\cfg\datasets\coco8.yaml"
    batch = 8
    epochs = 1
    imgsz = 640

    try:
        # 记录训练开始信息
        logger.info("=" * 50)
        logger.info("开始初始化YOLO模型...")
        logger.info(f"模型配置文件：{model_yaml}")
        logger.info(f"预训练权重：{pretrained_weights}")
        logger.info(f"batchsize：{batch}")
        logger.info(f"数据集配置：{data_yaml}")
        logger.info(f"训练轮数：{epochs}，输入图像尺寸：{imgsz}")
        logger.info("=" * 50)

        # 加载模型
        model = YOLO(model_yaml).load(pretrained_weights)
        logger.info("模型加载成功！")

        # 开始训练
        logger.info("启动模型训练...")
        start_time = datetime.now()
        results = model.train(
            data=data_yaml,
            batch=batch,
            epochs=epochs,
            imgsz=imgsz,
            # 可添加其他训练参数，如batch_size、lr0等
        )
        # 后续优化
        # print(results)
        end_time = datetime.now()
        train_duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟

        # 记录训练结束信息
        logger.info("=" * 50)
        logger.info("模型训练完成！")
        logger.info(f"训练耗时：{train_duration:.2f} 分钟")
        logger.info(f"最终训练指标：{results.results_dict}")
        logger.info(f"最佳模型保存路径：{results.save_dir}")
        logger.info("=" * 50)

        # 发送训练成功邮件
        email_subject = f"【YOLO训练完成】{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        email_content = f"""
        YOLO模型训练已成功完成！
        训练信息汇总：
        1. 模型：{model_yaml}
        2. 数据集：{data_yaml.split('/')[-1]}
        3. 训练轮数：{epochs}
        4. 训练耗时：{train_duration:.2f} 分钟
        5. 最佳模型路径：{results.save_dir}
        6. 最终数据：
        precision           {results.results_dict.get('metrics/precision(B)')}
        recall              {results.results_dict.get('metrics/recall(B)')}
        mAP50               {results.results_dict.get('metrics/mAP50(B)')}
        mAP50-95            {results.results_dict.get('metrics/mAP50-95(B)')}
        """
        send_train_notification(
            subject=email_subject,
            content=email_content,
            # 请替换为你的邮箱配置
            sender_email="xyyniao@163.com",
            sender_password="RVT2g337LNH4chD8",
            receiver_email="xyyniao@qq.com"
        )

    except Exception as e:
        # 记录训练异常
        logger.error("=" * 50)
        logger.error("训练过程中发生异常，训练终止！", exc_info=True)
        logger.error("=" * 50)

        # 发送训练失败邮件
        email_subject = f"【YOLO训练失败】{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        email_content = f"""
        YOLO模型训练过程中发生异常！
        异常信息：{str(e)}
        训练配置：
        模型：{model_yaml}
        数据集：{data_yaml}
        训练轮数：{epochs}
        """
        send_train_notification(
            subject=email_subject,
            content=email_content,
            # 请替换为你的邮箱配置
            sender_email="xyyniao@163.com",
            sender_password="RVT2g337LNH4chD8",
            receiver_email="xyyniao@qq.com"
        )