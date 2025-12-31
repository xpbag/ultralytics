# -*- coding: utf-8 -*-
"""
YOLO小目标数据增强脚本（集成ESRGAN超分+低噪声+随机遮挡）
核心更新：
1. 新增vis_compare开关，控制可视化模式（对比图/仅增强图）
2. 保留原有所有增强逻辑
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import warnings

# 忽略无关警告
warnings.filterwarnings('ignore')

# 导入ESRGAN所需依赖
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

# --------------------- 全局配置 ---------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()


# --------------------- ESRGAN模型定义 ---------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ESRGAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(ESRGAN, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# --------------------- 数据增强主类 ---------------------
class YOLOAugmenter:
    def __init__(self,
                 input_img_dir,
                 input_label_dir,
                 output_dir,
                 black_area_threshold=0.7,
                 max_workers=8,
                 augment_config=None,
                 esrgan_model_path=None,
                 global_aug_prob=0.5,
                 vis_compare=True):  # 新增：可视化模式开关（默认生成对比图）

        self.input_img_dir = Path(input_img_dir)
        self.input_label_dir = Path(input_label_dir)
        self.output_dir = Path(output_dir)
        self.black_threshold = black_area_threshold
        self.max_workers = max_workers
        self.global_aug_prob = global_aug_prob
        self.vis_compare = vis_compare  # 保存可视化模式开关

        # 创建输出目录
        self.aug_img_dir = self.output_dir / "augmented_images"
        self.aug_label_dir = self.output_dir / "augmented_labels"
        self.vis_dir = self.output_dir / "visualization"
        for dir_path in [self.aug_img_dir, self.aug_label_dir, self.vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 默认增强配置（新增per_obj_occlusion_prob：每个目标的遮挡概率）
        default_config = {
            "horizontal_flip": 0.5,
            "rotation": {"prob": 0.3, "angle_range": (-8, 8)},
            "brightness_contrast": {"prob": 0.4, "b_range": (-0.08, 0.08), "c_range": (-0.08, 0.08)},
            "noise": {"prob": 0.03, "noise_type": "gaussian", "intensity": (0, 2)},
            "scale_crop": {"prob": 0.3, "scale_range": (0.95, 1.2)},
            "small_obj_copy": {
                "prob": 0.2,
                "obj_size_thresh": 32 * 32,
                "max_copy": 1,
                "sr_prob": 0.7,
                "sr_scale": 2
            },
            "random_occlusion": {
                "prob": 0.1,  # 遮挡增强的全局触发概率
                "per_obj_occlusion_prob": 0.5,  # 新增：每个目标被遮挡的概率（50%）
                "occlusion_type": "partial",  # partial/full/random
                "occlusion_ratio": (0.1, 0.2),  # 遮挡区域占目标的比例
                "occlusion_color": (0, 0, 0)  # 遮挡颜色（黑色）
            }
        }

        # 合并自定义配置
        self.aug_config = default_config
        if augment_config is not None:
            for key in augment_config:
                if isinstance(augment_config[key], dict) and key in self.aug_config:
                    self.aug_config[key].update(augment_config[key])
                else:
                    self.aug_config[key] = augment_config[key]

        # 初始化ESRGAN模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.esrgan_model = self._init_esrgan(esrgan_model_path)

    def _init_esrgan(self, model_path):
        if model_path is None:
            model_path = "RRDB_ESRGAN_x4.pth"
        if not Path(model_path).exists():
            print(f"[提示] ESRGAN模型文件不存在：{model_path}")
            return None
        try:
            model = ESRGAN(nb=23)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model.to(self.device)
            print(f"[成功] ESRGAN模型加载完成，使用设备：{self.device}")
            return model
        except Exception as e:
            print(f"[错误] ESRGAN模型加载失败：{e}")
            return None

    def _esrgan_super_resolution(self, obj_patch, scale=2):
        if self.esrgan_model is None:
            h, w = obj_patch.shape[:2]
            obj_sr = cv2.resize(obj_patch, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            return obj_sr

        obj_patch = cv2.GaussianBlur(obj_patch, (1, 1), 0.5)
        img = ToTensor()(obj_patch).unsqueeze(0).to(self.device)
        img = img * 255.0

        with torch.no_grad():
            output = self.esrgan_model(img).data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = output.astype(np.uint8)

        h, w = obj_patch.shape[:2]
        target_h, target_w = int(h * scale), int(w * scale)
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return output

    def _calculate_black_area(self, img):
        black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))
        total_pixels = img.shape[0] * img.shape[1]
        return black_pixels / total_pixels if total_pixels > 0 else 0

    def _read_yolo_label(self, label_path, img_w, img_h):
        if not label_path.exists():
            return []
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    x_c = float(parts[1]) * img_w
                    y_c = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2
                    labels.append([cls, x1, y1, x2, y2])
                except:
                    continue
        return labels

    def _write_yolo_label(self, label_path, labels, img_w, img_h):
        with open(label_path, 'w', encoding='utf-8') as f:
            for label in labels:
                cls, x1, y1, x2, y2 = label
                x_c = (x1 + x2) / 2 / img_w
                y_c = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # --------------------- 基础增强函数 ---------------------
    def _aug_horizontal_flip(self, img, labels):
        img_flipped = cv2.flip(img, 1)
        img_w = img.shape[1]
        labels_flipped = []
        for cls, x1, y1, x2, y2 in labels:
            new_x1 = img_w - x2
            new_x2 = img_w - x1
            labels_flipped.append([cls, new_x1, y1, new_x2, y2])
        return img_flipped, labels_flipped, "horizontal_flip"

    def _aug_rotation(self, img, labels):
        cfg = self.aug_config["rotation"]
        angle = random.uniform(*cfg["angle_range"])
        img_h, img_w = img.shape[:2]
        center = (img_w // 2, img_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (img_w, img_h), borderValue=(0, 0, 0))
        img_rotated = cv2.GaussianBlur(img_rotated, (1, 1), 0.3)

        labels_rotated = []
        for cls, x1, y1, x2, y2 in labels:
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            points = np.hstack([points, np.ones((4, 1))])
            points_rotated = np.dot(M, points.T).T
            new_x1 = np.min(points_rotated[:, 0])
            new_y1 = np.min(points_rotated[:, 1])
            new_x2 = np.max(points_rotated[:, 0])
            new_y2 = np.max(points_rotated[:, 1])
            if new_x1 >= 0 and new_y1 >= 0 and new_x2 <= img_w and new_y2 <= img_h:
                labels_rotated.append([cls, new_x1, new_y1, new_x2, new_y2])
        return img_rotated, labels_rotated, f"rotation_{angle:.1f}"

    def _aug_brightness_contrast(self, img, labels):
        cfg = self.aug_config["brightness_contrast"]
        b_delta = random.uniform(*cfg["b_range"])
        c_delta = random.uniform(*cfg["c_range"])
        img_aug = cv2.convertScaleAbs(img, alpha=1 + c_delta, beta=b_delta * 255)
        return img_aug, labels, f"brightness_{b_delta:.2f}_contrast_{c_delta:.2f}"

    def _aug_noise(self, img, labels):
        cfg = self.aug_config["noise"]
        noise_type = cfg["noise_type"]
        intensity = random.uniform(*cfg["intensity"])

        if noise_type == "gaussian":
            mask = np.all(img != [0, 0, 0], axis=-1).astype(np.uint8)
            noise = np.random.normal(0, intensity, img.shape).astype(np.uint8)
            noise = noise * mask[:, :, np.newaxis]
            img_noisy = cv2.add(img, noise)
            return img_noisy, labels, f"{noise_type}_noise_{intensity:.1f}"
        return img, labels, "no_noise"

    def _aug_scale_crop(self, img, labels):
        cfg = self.aug_config["scale_crop"]
        scale = random.uniform(*cfg["scale_range"])
        img_h, img_w = img.shape[:2]
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        labels_scaled = []
        if scale > 1.0:
            x_start = (new_w - img_w) // 2
            y_start = (new_h - img_h) // 2
            img_cropped = img_scaled[y_start:y_start + img_h, x_start:x_start + img_w]
            for cls, x1, y1, x2, y2 in labels:
                new_x1 = x1 * scale - x_start
                new_y1 = y1 * scale - y_start
                new_x2 = x2 * scale - x_start
                new_y2 = y2 * scale - y_start
                if new_x1 >= 0 and new_y1 >= 0 and new_x2 <= img_w and new_y2 <= img_h:
                    labels_scaled.append([cls, new_x1, new_y1, new_x2, new_y2])
        else:
            img_cropped = np.zeros_like(img)
            x_start = (img_w - new_w) // 2
            y_start = (img_h - new_h) // 2
            img_cropped[y_start:y_start + new_h, x_start:x_start + new_w] = img_scaled
            for cls, x1, y1, x2, y2 in labels:
                new_x1 = x1 * scale + x_start
                new_y1 = y1 * scale + y_start
                new_x2 = x2 * scale + x_start
                new_y2 = y2 * scale + y_start
                labels_scaled.append([cls, new_x1, new_y1, new_x2, new_y2])
        return img_cropped, labels_scaled, f"scale_{scale:.2f}"

    def _aug_small_obj_copy(self, img, labels):
        cfg = self.aug_config["small_obj_copy"]
        obj_size_thresh = cfg["obj_size_thresh"]
        max_copy = cfg["max_copy"]
        sr_prob = cfg["sr_prob"]
        sr_scale = cfg["sr_scale"]

        small_objs = []
        for label in labels:
            cls, x1, y1, x2, y2 = label
            obj_area = (x2 - x1) * (y2 - y1)
            if obj_area < obj_size_thresh:
                small_objs.append(label)

        if not small_objs:
            return img, labels, "no_small_obj_copy"

        copy_num = random.randint(1, min(max_copy, len(small_objs)))
        img_copy = img.copy()
        labels_copy = labels.copy()
        img_h, img_w = img.shape[:2]
        sr_flag = False

        for _ in range(copy_num):
            obj = random.choice(small_objs)
            cls, x1, y1, x2, y2 = obj
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            obj_w, obj_h = x2_int - x1_int, y2_int - y1_int

            if obj_w <= 0 or obj_h <= 0:
                continue

            obj_patch = img[y1_int:y2_int, x1_int:x2_int]
            if random.random() < sr_prob:
                obj_patch = self._esrgan_super_resolution(obj_patch, sr_scale)
                obj_h_new, obj_w_new = obj_patch.shape[:2]
                obj_w, obj_h = obj_w_new, obj_h_new
                sr_flag = True

            new_x1 = random.randint(0, img_w - obj_w)
            new_y1 = random.randint(0, img_h - obj_h)
            new_x2 = new_x1 + obj_w
            new_y2 = new_y1 + obj_h

            img_copy[new_y1:new_y2, new_x1:new_x2] = obj_patch
            labels_copy.append([cls, new_x1, new_y1, new_x2, new_y2])

        sr_suffix = "_esrgan_sr" if sr_flag else ""
        aug_name = f"small_obj_copy_{copy_num}{sr_suffix}"
        return img_copy, labels_copy, aug_name

    # --------------------- 核心修改：随机遮挡增强（支持每个目标独立概率） ---------------------
    def _aug_random_occlusion(self, img, labels):
        """
        随机遮挡增强（修改后）：
        1. 先判断遮挡增强是否全局触发（prob）
        2. 对每个目标单独判断是否遮挡（per_obj_occlusion_prob）
        3. 支持partial/full遮挡模式，适配所有目标
        """
        cfg = self.aug_config["random_occlusion"]
        occlusion_type = cfg["occlusion_type"]
        occlusion_ratio = random.uniform(*cfg["occlusion_ratio"])
        occlusion_color = cfg["occlusion_color"]
        per_obj_prob = cfg["per_obj_occlusion_prob"]  # 每个目标的遮挡概率
        img_occluded = img.copy()
        img_h, img_w = img.shape[:2]
        occluded_count = 0  # 统计被遮挡的目标数

        # 遍历所有目标，每个目标独立判断是否遮挡
        for label in labels:
            cls, x1, y1, x2, y2 = label
            obj_w = x2 - x1
            obj_h = y2 - y1

            # 跳过无效目标
            if obj_w <= 0 or obj_h <= 0:
                continue

            # 核心：每个目标单独判断是否执行遮挡
            if random.random() < per_obj_prob:
                occluded_count += 1

                if occlusion_type == "partial":
                    # 局部遮挡：遮挡目标的一部分区域
                    occl_w = int(obj_w * occlusion_ratio)
                    occl_h = int(obj_h * occlusion_ratio)
                    # 随机选择遮挡位置（目标内部）
                    occl_x1 = int(x1 + random.uniform(0, max(1, obj_w - occl_w)))
                    occl_y1 = int(y1 + random.uniform(0, max(1, obj_h - occl_h)))
                    occl_x2 = occl_x1 + occl_w
                    occl_y2 = occl_y1 + occl_h
                    # 绘制遮挡区域
                    img_occluded[occl_y1:occl_y2, occl_x1:occl_x2] = occlusion_color

                elif occlusion_type == "full":
                    # 完全遮挡：覆盖整个目标
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    img_occluded[y1_int:y2_int, x1_int:x2_int] = occlusion_color

        # 随机区域遮挡（独立于目标遮挡）
        if occlusion_type == "random" and random.random() < cfg["prob"]:
            occl_w = int(img_w * occlusion_ratio)
            occl_h = int(img_h * occlusion_ratio)
            occl_x1 = random.randint(0, max(0, img_w - occl_w))
            occl_y1 = random.randint(0, max(0, img_h - occl_h))
            occl_x2 = occl_x1 + occl_w
            occl_y2 = occl_y1 + occl_h
            img_occluded[occl_y1:occl_y2, occl_x1:occl_x2] = occlusion_color
            occluded_count = -1  # 标记为随机区域遮挡

        # 生成增强标识
        if occluded_count > 0:
            aug_name = f"random_occlusion_{occlusion_type}_obj{occluded_count}_ratio{occlusion_ratio:.2f}"
        elif occluded_count == -1:
            aug_name = f"random_occlusion_random_ratio{occlusion_ratio:.2f}"
        else:
            aug_name = "no_occlusion"

        return img_occluded, labels, aug_name

    # --------------------- 可视化与处理函数 ---------------------
    def _draw_labels(self, img, labels):
        img_draw = img.copy()
        for cls, x1, y1, x2, y2 in labels:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img_draw

    def _generate_vis(self, img_ori, labels_ori, img_aug, labels_aug, vis_path, aug_name):
        try:
            if img_aug is None:  # 仅校验增强图是否存在
                print(f"[警告] 可视化失败：增强图片为空 - {vis_path}")
                return

            # 预处理增强图
            img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)
            img_aug = cv2.GaussianBlur(img_aug, (1, 1), 0.3)
            try:
                img_aug_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            except:
                img_aug_rgb = img_aug
            img_aug_draw = self._draw_labels(img_aug_rgb, labels_aug)

            # 根据vis_compare开关分支处理
            if self.vis_compare:
                # 模式1：生成原图+增强图对比（原有逻辑）
                if img_ori is None:
                    print(f"[警告] 可视化失败：原图为空 - {vis_path}")
                    return
                img_ori = np.clip(img_ori, 0, 255).astype(np.uint8)
                img_ori = cv2.GaussianBlur(img_ori, (1, 1), 0.3)
                try:
                    img_ori_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                except:
                    img_ori_rgb = img_ori
                img_ori_draw = self._draw_labels(img_ori_rgb, labels_ori)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.imshow(img_ori_draw)
                ax1.set_title("增强前", fontsize=14)
                ax1.axis('off')
                ax2.imshow(img_aug_draw)
                ax2.set_title(f"增强后: {aug_name}", fontsize=14)
                ax2.axis('off')
            else:
                # 模式2：仅生成增强图（新逻辑）
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(img_aug_draw)
                ax.set_title(f"增强结果: {aug_name}", fontsize=14)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(str(vis_path), dpi=100, bbox_inches=None)
            plt.close(fig)
        except Exception as e:
            print(f"[错误] 可视化失败 {vis_path}：{e}")
            plt.close('all')

    def process_single_image(self, img_name):
        try:
            # 全局增强概率判断
            if random.random() > self.global_aug_prob:
                print(f"[跳过] {img_name} - 全局增强概率未触发")
                return None

            # 读取图片和标签
            img_path = self.input_img_dir / img_name
            label_name = Path(img_name).stem + ".txt"
            label_path = self.input_label_dir / label_name

            img_ori = cv2.imread(str(img_path))
            if img_ori is None:
                print(f"[警告] 无法读取图片: {img_name}")
                return None

            img_h, img_w = img_ori.shape[:2]
            labels_ori = self._read_yolo_label(label_path, img_w, img_h)

            # 过滤黑边过多的图片
            black_ratio = self._calculate_black_area(img_ori)
            if black_ratio > self.black_threshold:
                print(f"[跳过] {img_name} - 黑边占比{black_ratio:.2%} > {self.black_threshold:.0%}")
                return None

            # 执行增强操作
            img_aug = img_ori.copy()
            labels_aug = labels_ori.copy()
            aug_names = []

            # 水平翻转
            if random.random() < self.aug_config["horizontal_flip"]:
                img_aug, labels_aug, aug_name = self._aug_horizontal_flip(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 旋转
            if random.random() < self.aug_config["rotation"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_rotation(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 亮度对比度
            if random.random() < self.aug_config["brightness_contrast"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_brightness_contrast(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 噪声
            if random.random() < self.aug_config["noise"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_noise(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 缩放裁剪
            if random.random() < self.aug_config["scale_crop"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_scale_crop(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 小目标复制超分
            if random.random() < self.aug_config["small_obj_copy"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_small_obj_copy(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 随机遮挡（支持每个目标独立概率）
            if random.random() < self.aug_config["random_occlusion"]["prob"]:
                img_aug, labels_aug, aug_name = self._aug_random_occlusion(img_aug, labels_aug)
                aug_names.append(aug_name)

            # 无增强触发则跳过保存
            if len(aug_names) == 0:
                print(f"[跳过] {img_name} - 无增强操作触发，不重复保存")
                return None

            # 保存增强后的图片和标签
            aug_suffix = "_".join(aug_names)
            stem = Path(img_name).stem
            ext = Path(img_name).suffix
            aug_img_name = f"{stem}_{aug_suffix}{ext}"
            aug_label_name = f"{stem}_{aug_suffix}.txt"

            if len(labels_aug) > 0:
                cv2.imwrite(str(self.aug_img_dir / aug_img_name), img_aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self._write_yolo_label(self.aug_label_dir / aug_label_name, labels_aug, img_w, img_h)
                print(f"[完成] {img_name} -> {aug_img_name}")

                # 返回可视化参数
                vis_path = self.vis_dir / f"{stem}_aug_comparison.jpg"
                return (img_ori, labels_ori, img_aug, labels_aug, vis_path, aug_suffix)
            else:
                print(f"[跳过] {img_name} - 增强后无有效标签")
                return None
        except Exception as e:
            print(f"[错误] 处理{img_name}失败: {e}")
            return None

    def process_all(self):
        # 筛选图片文件
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        img_files = [f for f in os.listdir(self.input_img_dir)
                     if Path(f).suffix.lower() in img_extensions]

        if not img_files:
            print("[错误] 未找到任何图片文件")
            return

        print(f"[开始] 共发现{len(img_files)}张图片，全局增强概率{self.global_aug_prob}，线程数{self.max_workers}")
        print(f"[可视化模式] {'对比图模式' if self.vis_compare else '仅增强图模式'}")

        # 多线程处理
        aug_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_image, f) for f in img_files]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    aug_results.append(res)

        # 生成可视化
        print(f"[开始] 生成可视化图（共{len(aug_results)}张）")
        for res in aug_results:
            self._generate_vis(*res)

        # 输出结果
        print(f"[完成] 增强结果保存至: {self.output_dir}")
        print(f"  - 增强图片: {self.aug_img_dir}（共{len(aug_results)}张）")
        print(f"  - 增强标签: {self.aug_label_dir}")
        print(f"  - 可视化图: {self.vis_dir}")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 配置参数
    INPUT_IMG_DIR = r"D:\Dataset\AA_VisDrone\images"
    INPUT_LABEL_DIR = r"D:\Dataset\AA_VisDrone\labels"
    OUTPUT_DIR = r"D:\Dataset\AA_VisDrone_Augmented"
    BLACK_THRESHOLD = 0.7
    MAX_WORKERS = 8 if torch.cuda.is_available() else 16
    ESRGAN_MODEL_PATH = r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\utils\RRDB_ESRGAN_x4.pth"
    GLOBAL_AUG_PROB = 0.3
    VIS_COMPARE = False  # 核心开关：False=仅增强图，True=对比图

    # 自定义增强配置（可调整每个目标的遮挡概率）
    CUSTOM_AUG_CONFIG = {
        "horizontal_flip": 0.5,
        "rotation": {"prob": 0.3, "angle_range": (-10, 10)},
        "brightness_contrast": {"prob": 0.4, "b_range": (-0.08, 0.08), "c_range": (-0.08, 0.08)},
        "noise": {"prob": 0.0, "noise_type": "gaussian", "intensity": (0, 2)},
        "scale_crop": {"prob": 0.3, "scale_range": (0.95, 1.2)},
        "small_obj_copy": {
            "prob": 0.2,
            "obj_size_thresh": 32 * 32,
            "max_copy": 1,
            "sr_prob": 0.7,
            "sr_scale": 2
        },
        "random_occlusion": {
            "prob": 0.2,  # 遮挡增强的全局触发概率（20%）
            "per_obj_occlusion_prob": 0.6,  # 每个目标被遮挡的概率（60%）
            "occlusion_type": "partial",  # 局部遮挡所有符合概率的目标
            "occlusion_ratio": (0.1, 0.3),  # 遮挡区域占目标的10%-30%
            "occlusion_color": (50, 50, 50)  # 灰色遮挡（更贴近真实场景）
        }
    }

    # 创建增强器并执行（传入可视化模式开关）
    augmenter = YOLOAugmenter(
        input_img_dir=INPUT_IMG_DIR,
        input_label_dir=INPUT_LABEL_DIR,
        output_dir=OUTPUT_DIR,
        black_area_threshold=BLACK_THRESHOLD,
        max_workers=MAX_WORKERS,
        augment_config=CUSTOM_AUG_CONFIG,
        esrgan_model_path=ESRGAN_MODEL_PATH,
        global_aug_prob=GLOBAL_AUG_PROB,
        vis_compare=VIS_COMPARE  # 传入开关参数
    )
    augmenter.process_all()