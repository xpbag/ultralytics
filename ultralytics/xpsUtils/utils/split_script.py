import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，避免可视化时中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 禁用Matplotlib交互模式，提升保存速度
plt.ioff()


class YOLOImageSplitter:
    def __init__(self, img_dir, label_dir, output_dir, tile_size=640, area_threshold=0.3, max_workers=4):
        """
        初始化YOLO图片切割器（优化版）
        :param img_dir: 原始图片目录
        :param label_dir: 原始YOLO标签目录
        :param output_dir: 输出目录（存放切割后的图片和标签）
        :param tile_size: 切割后的图片尺寸，默认640x640
        :param area_threshold: 目标可见面积占比阈值，默认0.3（30%）
        :param max_workers: 最大线程数，默认4（根据CPU核心数调整）
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.area_threshold = area_threshold
        self.max_workers = max_workers

        # 创建输出目录（提前创建，避免重复判断）
        self.output_img_dir = self.output_dir / "images"
        self.output_label_dir = self.output_dir / "labels"
        self.output_vis_dir = self.output_dir / "visualization"
        self.output_tile_vis_dir = self.output_dir / "tile_visualization"

        for dir_path in [self.output_img_dir, self.output_label_dir, self.output_vis_dir, self.output_tile_vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 预定义常量，避免重复计算
        self.tile_size_float = float(self.tile_size)

    def read_yolo_label(self, label_path, img_width, img_height):
        """
        快速读取YOLO格式标签（优化IO和计算）
        :return: 列表，每个元素为 (class_id, x1, y1, x2, y2, area)
        """
        if not label_path.exists():
            return []

        # 一次性读取所有行，提升IO速度
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except:
            return []

        labels = []
        # 预计算宽高浮点数，避免重复转换
        img_w = float(img_width)
        img_h = float(img_height)

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h

                # 计算边界框坐标
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # 计算面积（提前过滤0面积）
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    continue

                labels.append((class_id, x1, y1, x2, y2, area))
            except:
                continue

        return labels

    def calculate_overlap_area(self, obj_rect, tile_rect):
        """
        快速计算交叠面积（减少变量创建）
        :param obj_rect: (x1, y1, x2, y2) 目标矩形
        :param tile_rect: (x1, y1, x2, y2) 切割块矩形
        :return: 交叠面积
        """
        # 直接解包计算，减少中间变量
        x1 = max(obj_rect[0], tile_rect[0])
        y1 = max(obj_rect[1], tile_rect[1])
        x2 = min(obj_rect[2], tile_rect[2])
        y2 = min(obj_rect[3], tile_rect[3])

        return (x2 - x1) * (y2 - y1) if x1 < x2 and y1 < y2 else 0.0

    def write_yolo_label(self, label_path, labels):
        """
        快速写入YOLO标签（预计算常量）
        """
        if not labels:
            return

        try:
            with open(label_path, 'w', encoding='utf-8') as f:
                lines = []
                for label in labels:
                    class_id, x1, y1, x2, y2 = label

                    # 使用预计算的常量，减少重复计算
                    x_center = (x1 + x2) / 2 / self.tile_size_float
                    y_center = (y1 + y2) / 2 / self.tile_size_float
                    width = (x2 - x1) / self.tile_size_float
                    height = (y2 - y1) / self.tile_size_float

                    # 限制范围并格式化
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))

                    lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # 一次性写入，提升IO速度
                f.write('\n'.join(lines))
        except:
            pass

    def draw_labels_fast(self, img, labels):
        """
        快速绘制标签（减少重复操作）
        """
        img_copy = img.copy()
        for label in labels:
            try:
                class_id, x1, y1, x2, y2 = label
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 一次性绘制矩形和文字
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_copy, str(class_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                continue
        return img_copy

    def process_single_image(self, img_file):
        """
        处理单张图片（供多线程调用）
        """
        try:
            img_path = self.img_dir / img_file
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[警告] 无法读取图片：{img_file}")
                return 0

            # 预计算图片尺寸
            img_height, img_width = img.shape[:2]
            label_name = Path(img_file).stem + ".txt"
            label_path = self.label_dir / label_name

            # 读取标签
            original_labels = self.read_yolo_label(label_path, img_width, img_height)

            # 存储有效切割块信息
            tile_info = []
            valid_tile_count = 0
            tile_idx = 0

            # 预计算滑动窗口的步数
            y_steps = range(0, img_height, self.tile_size)
            x_steps = range(0, img_width, self.tile_size)

            for y in y_steps:
                y_start = y
                y_end = min(y + self.tile_size, img_height)

                for x in x_steps:
                    x_start = x
                    x_end = min(x + self.tile_size, img_width)
                    tile_rect = (x_start, y_start, x_end, y_end)

                    # 快速提取切割块
                    tile = img[y_start:y_end, x_start:x_end]
                    tile_h, tile_w = tile.shape[:2]

                    # 补黑边（使用预分配数组）
                    if tile_w < self.tile_size or tile_h < self.tile_size:
                        new_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                        new_tile[:tile_h, :tile_w] = tile
                        tile = new_tile

                    # 筛选有效目标
                    tile_labels = []
                    for label in original_labels:
                        class_id, lx1, ly1, lx2, ly2, total_area = label
                        obj_rect = (lx1, ly1, lx2, ly2)

                        # 计算交叠面积和占比
                        overlap_area = self.calculate_overlap_area(obj_rect, tile_rect)
                        area_ratio = overlap_area / total_area

                        if area_ratio > self.area_threshold:
                            # 计算切割块内坐标
                            tx1 = max(0.0, lx1 - x_start)
                            ty1 = max(0.0, ly1 - y_start)
                            tx2 = min(self.tile_size_float, lx2 - x_start)
                            ty2 = min(self.tile_size_float, ly2 - y_start)

                            tile_labels.append((class_id, tx1, ty1, tx2, ty2))

                    # 仅保存有有效目标的切割块
                    if tile_labels:
                        # 构建文件名（提前格式化）
                        stem = Path(img_file).stem
                        tile_img_name = f"{stem}_{tile_idx}.jpg"
                        tile_label_name = f"{stem}_{tile_idx}.txt"

                        # 保存图片（禁用压缩，提升速度）
                        tile_img_path = self.output_img_dir / tile_img_name
                        cv2.imwrite(str(tile_img_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        # 保存标签
                        tile_label_path = self.output_label_dir / tile_label_name
                        self.write_yolo_label(tile_label_path, tile_labels)

                        # 保存可视化（跳过压缩，提升速度）
                        tile_vis_path = self.output_tile_vis_dir / tile_img_name
                        tile_with_labels = self.draw_labels_fast(tile, tile_labels)
                        cv2.imwrite(str(tile_vis_path), tile_with_labels, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        # 记录信息
                        tile_info.append({
                            'x': x_start,
                            'y': y_start,
                            'width': tile_w,
                            'height': tile_h,
                            'img_path': tile_img_path,
                            'labels': tile_labels
                        })
                        valid_tile_count += 1

                    tile_idx += 1

            # 生成拼接验证图（可选，如需进一步提速可注释）
            if tile_info:
                self.generate_verification_mosaic(img_path, img, original_labels, tile_info)

            print(f"[完成] {img_file} - 总块数：{tile_idx} | 有效块数：{valid_tile_count}")
            return valid_tile_count

        except Exception as e:
            print(f"[错误] 处理 {img_file} 失败：{str(e)}")
            return 0

    def generate_verification_mosaic(self, img_path, original_img, original_labels, tile_info):
        """
        快速生成拼接验证图
        """
        try:
            img_name = Path(img_path).name
            img_height, img_width = original_img.shape[:2]

            # 绘制原图标签
            original_labels_for_draw = [(l[0], l[1], l[2], l[3], l[4]) for l in original_labels]
            original_img_with_labels = self.draw_labels_fast(original_img, original_labels_for_draw)

            # 创建拼接画布
            mosaic = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # 快速拼接
            for tile in tile_info:
                x, y = tile['x'], tile['y']
                tile_w, tile_h = tile['width'], tile['height']
                tile_img = cv2.imread(str(tile['img_path']))

                if tile_img is not None:
                    tile_with_labels = self.draw_labels_fast(tile_img, tile['labels'])
                    mosaic[y:y + tile_h, x:x + tile_w] = tile_with_labels[:tile_h, :tile_w]

            # 生成对比图（使用快速设置）
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
            ax1.imshow(cv2.cvtColor(original_img_with_labels, cv2.COLOR_BGR2RGB))
            ax1.set_title('原始图片 + 原始标签', fontsize=16)
            ax1.axis('off')

            ax2.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'有效切割块拼接（面积阈值{self.area_threshold:.0%}）', fontsize=16)
            ax2.axis('off')

            # 保存图片（禁用压缩）
            vis_path = self.output_vis_dir / f"{Path(img_name).stem}_comparison.jpg"
            plt.savefig(str(vis_path), dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        except:
            pass

    def process_all(self):
        """
        多线程批量处理所有图片（核心提速）
        """
        # 获取所有图片文件（过滤常见格式）
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        img_files = [
            f for f in os.listdir(self.img_dir)
            if Path(f).suffix.lower() in img_extensions
        ]

        if not img_files:
            print("[错误] 未找到任何图片文件！")
            return

        print(f"\n[开始处理] 共发现 {len(img_files)} 张图片")
        print(f"[配置] 切割尺寸：{self.tile_size} | 面积阈值：{self.area_threshold:.0%} | 线程数：{self.max_workers}")
        print("-" * 60)

        # 多线程处理
        total_valid = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_img = {executor.submit(self.process_single_image, img_file): img_file for img_file in img_files}

            # 收集结果
            for future in as_completed(future_to_img):
                try:
                    total_valid += future.result()
                except:
                    continue

        # 输出汇总
        print("-" * 60)
        print(f"\n[处理完成] 汇总：")
        print(f"  原始图片总数：{len(img_files)}")
        print(f"  生成有效切割块总数：{total_valid}")
        print(f"  输出目录：{self.output_dir}")
        print(f"  - 切割图片：{self.output_img_dir}")
        print(f"  - 标签文件：{self.output_label_dir}")
        print(f"  - 单块可视化：{self.output_tile_vis_dir}")
        print(f"  - 拼接验证图：{self.output_vis_dir}")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 配置参数（已适配你的路径）
    IMG_DIR = r"D:\Dataset\VisDrone2019-DET\train\images"  # 原始图片目录
    LABEL_DIR = r"D:\Dataset\VisDrone2019-DET\train\labels"  # 原始YOLO标签目录
    OUTPUT_DIR = r"D:\Dataset\AA_VisDrone"  # 输出目录
    TILE_SIZE = 640  # 切割尺寸
    AREA_THRESHOLD = 0.3  # 面积占比阈值
    MAX_WORKERS = 8  # 线程数（建议设为CPU核心数）

    # 创建切割器实例
    splitter = YOLOImageSplitter(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        tile_size=TILE_SIZE,
        area_threshold=AREA_THRESHOLD,
        max_workers=MAX_WORKERS
    )

    # 批量处理所有图片
    splitter.process_all()