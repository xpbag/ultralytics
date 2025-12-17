from ultralytics import YOLO
import logging
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import queue
import threading
import time
import sys
import io
from typing import Optional

# -------------------------- 全局变量 --------------------------
# 保存原始的文件描述符
ORIGINAL_STDOUT_FD = sys.stdout.fileno()
ORIGINAL_STDERR_FD = sys.stderr.fileno()


# -------------------------- 异步日志处理器 --------------------------
class AsyncFileHandler(logging.Handler):
    """异步文件日志处理器，支持高并发写入"""

    def __init__(self, filename: str, encoding: str = "utf-8", max_queue_size: int = 100000):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()
        self.file_handle: Optional[io.TextIOWrapper] = None
        self.lock = threading.Lock()

    def _init_file_handle(self):
        """线程安全的文件句柄初始化"""
        with self.lock:
            if self.file_handle is None:
                self.file_handle = open(self.filename, "a", encoding=self.encoding, buffering=1)

    def _write_loop(self):
        """异步写入循环，确保所有消息都被写入"""
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                record = self.queue.get(timeout=2)
                self._init_file_handle()
                if self.file_handle:
                    msg = self.format(record)
                    self.file_handle.write(f"{msg}\n")
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"异步日志写入错误：{e}")

        # 清理资源
        with self.lock:
            if self.file_handle:
                self.file_handle.flush()
                self.file_handle.close()
                self.file_handle = None

    def emit(self, record):
        """将日志记录放入队列"""
        try:
            if self.queue.full():
                # 队列满时丢弃最老的消息，避免阻塞
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)

    def close(self):
        """优雅关闭"""
        self._stop_event.set()
        self._write_thread.join(timeout=10)
        super().close()


# -------------------------- 文件描述符重定向类 --------------------------
class FDRedirector:
    """重定向进程的文件描述符，捕获所有层级的输出"""

    def __init__(self, logger, stdout_level=logging.INFO, stderr_level=logging.ERROR):
        self.logger = logger
        self.stdout_level = stdout_level
        self.stderr_level = stderr_level
        self.stdout_r, self.stdout_w = os.pipe()
        self.stderr_r, self.stderr_w = os.pipe()
        self._stop_event = threading.Event()
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)

    def start(self):
        """启动重定向"""
        # 保存原始的fd
        self.original_stdout = os.dup(ORIGINAL_STDOUT_FD)
        self.original_stderr = os.dup(ORIGINAL_STDERR_FD)

        # 重定向stdout/stderr到管道
        os.dup2(self.stdout_w, ORIGINAL_STDOUT_FD)
        os.dup2(self.stderr_w, ORIGINAL_STDERR_FD)

        # 启动读取线程
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        """读取stdout管道内容"""
        with os.fdopen(self.stdout_r, 'r', encoding='utf-8', errors='ignore') as f:
            while not self._stop_event.is_set():
                line = f.readline()
                if line:
                    self.logger.log(self.stdout_level, line.rstrip('\n'))
                else:
                    time.sleep(0.001)

    def _read_stderr(self):
        """读取stderr管道内容"""
        with os.fdopen(self.stderr_r, 'r', encoding='utf-8', errors='ignore') as f:
            while not self._stop_event.is_set():
                line = f.readline()
                if line:
                    self.logger.log(self.stderr_level, line.rstrip('\n'))
                else:
                    time.sleep(0.001)

    def stop(self):
        """停止重定向并恢复"""
        self._stop_event.set()

        # 恢复原始的fd
        os.dup2(self.original_stdout, ORIGINAL_STDOUT_FD)
        os.dup2(self.original_stderr, ORIGINAL_STDERR_FD)

        # 关闭管道
        os.close(self.stdout_r)
        os.close(self.stdout_w)
        os.close(self.stderr_r)
        os.close(self.stderr_w)

        # 等待线程结束
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)


# -------------------------- 日志配置函数 --------------------------
def setup_logger(log_dir="./yolo_train_logs"):
    """
    配置全量日志捕获：
    1. 异步文件日志（DEBUG级别）
    2. 控制台输出（INFO级别）
    3. 文件描述符级别的stdout/stderr重定向
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"yolo11s_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 创建日志器
    logger = logging.getLogger("YOLO_Train_Logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    # 日志格式（包含详细时间和级别）
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 异步文件处理器
    async_file_handler = AsyncFileHandler(log_file, encoding="utf-8")
    async_file_handler.setLevel(logging.DEBUG)
    async_file_handler.setFormatter(formatter)
    logger.addHandler(async_file_handler)

    # 初始化文件描述符重定向
    fd_redirector = FDRedirector(logger)
    fd_redirector.start()

    return logger, fd_redirector


# -------------------------- 邮件提醒函数 --------------------------
def send_train_notification(
        subject, content,
        sender_email="xyyniao@163.com",
        sender_password="RVT2g337LNH4chD8",
        receiver_email="xyyniao@qq.com",
        smtp_server="smtp.163.com",
        smtp_port=465
):
    """发送训练通知邮件"""
    logger = logging.getLogger("YOLO_Train_Logger")
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(content, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email.split(","), msg.as_string())
        logger.info("训练提醒邮件发送成功！")
    except Exception as e:
        logger.error(f"邮件发送失败：{str(e)}", exc_info=True)


# -------------------------- 主训练流程 --------------------------
if __name__ == '__main__':
    # 初始化日志（含文件描述符重定向）
    logger, fd_redirector = setup_logger()

    # 记录程序启动信息
    logger.info(f"程序启动命令：{' '.join(sys.argv)}")
    logger.info(f"Python解释器路径：{sys.executable}")

    # 训练配置
    model_yaml = "yolo11n.yaml"
    pretrained_weights = "D:\\learningJournal\\Detection\\ultralytics\\ultralytics\\xpsUtils\\yolo11n.pt"
    data_yaml = "D:\\learningJournal\\Detection\\ultralytics\\ultralytics\\cfg\\datasets\\coco8.yaml"
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
            verbose=True,  # 强制输出详细日志
            plots=True
        )
        end_time = datetime.now()
        train_duration = (end_time - start_time).total_seconds() / 60

        # 记录训练结束信息
        logger.info("=" * 50)
        logger.info("模型训练完成！")
        logger.info(f"训练耗时：{train_duration:.2f} 分钟")
        logger.info(f"最终训练指标：{results.results_dict}")
        logger.info(f"最佳模型保存路径：{results.save_dir}")
        logger.info("=" * 50)

        # 处理数据集名称
        data_name = os.path.basename(data_yaml)

        # 发送成功邮件
        email_subject = f"【YOLO训练完成】{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        email_content = f"""
        YOLO模型训练已成功完成！
        训练信息汇总：
        1. 模型：{model_yaml}
        2. 数据集：{data_name}
        3. 训练轮数：{epochs}
        4. 训练耗时：{train_duration:.2f} 分钟
        5. 最佳模型路径：{results.save_dir}
        6. 最终mAP50：{results.results_dict.get('mAP50(B)', 'N/A')}
        """
        send_train_notification(
            subject=email_subject,
            content=email_content,
            sender_email="xyyniao@163.com",
            sender_password="RVT2g337LNH4chD8",
            receiver_email="xyyniao@qq.com"
        )

    except Exception as e:
        # 记录异常
        logger.error("=" * 50)
        logger.error("训练过程中发生异常，训练终止！", exc_info=True)
        logger.error("=" * 50)

        # 发送失败邮件
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
            sender_email="xyyniao@163.com",
            sender_password="RVT2g337LNH4chD8",
            receiver_email="xyyniao@qq.com"
        )
    finally:
        # 优雅清理资源
        logger.info("开始清理资源...")

        # 停止文件描述符重定向
        fd_redirector.stop()

        # 关闭日志处理器
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        # 等待异步线程处理完剩余日志
        time.sleep(3)
        logger.info("程序正常退出！")