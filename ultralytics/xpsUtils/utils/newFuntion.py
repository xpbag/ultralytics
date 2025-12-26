import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter


# 配置参数
class Config:
    # 原始数据路径（用户指定路径）
    INPUT_IMG_DIR = r"D:\Dataset\AmyNewDateSet\val\images"  # 原始图片文件夹
    INPUT_LABEL_DIR = r"D:\Dataset\AmyNewDateSet\val\labels"  # 原始标签文件夹
    # 增强后数据路径
    OUTPUT_IMG_DIR = r"D:\Dataset\AmyNewDateSet\test\images"  # 增强后图片文件夹
    OUTPUT_LABEL_DIR = r"D:\Dataset\AmyNewDateSet\test\labels"  # 增强后标签文件夹
    # 可视化路径
    VISUAL_DIR = r"D:\Dataset\AmyNewDateSet\eyes"  # 可视化结果文件夹

    # 基础增强参数（降低噪声相关强度）
    SCALE_RANGE = (0.7, 1.3)  # 随机缩放范围
    CROP_RATIO_RANGE = (0.6, 0.95)  # 随机裁剪比例范围（相对于原图宽高）
    ROTATE_RANGE = (-30, 30)  # 随机旋转角度范围
    TRANSLATE_RANGE = (-0.1, 0.1)  # 随机平移范围（相对尺寸）
    BRIGHTNESS_RANGE = (0.7, 1.3)  # 亮度调整范围（收窄，减少过度变化）
    CONTRAST_RANGE = (0.7, 1.3)  # 对比度调整范围（收窄）
    NOISE_PROB = 0.3  # 噪声添加概率（从0.5降低到0.3，减少噪声出现频率）
    GAUSSIAN_NOISE_SIGMA = (0, 10)  # 高斯噪声标准差范围（从0-20降低到0-10，减弱噪声）
    SALT_PEPPER_NOISE_RATIO = 0.005  # 椒盐噪声比例（从0.02降低到0.005，减少噪点数量）

    # 小目标增强参数
    SMALL_OBJECT_THRESH = 0.01  # 小目标的判定阈值（目标面积占图片总面积的比例＜1%）
    OVERSAMPLING_FACTOR = 2  # 小目标过采样倍数
    SYNTHESIS_NUM = 3  # 每张图合成的小目标数量
    SYNTHESIS_SCALE_RANGE = (0.8, 1.2)  # 合成小目标的缩放比例（避免尺寸单一）
    # 超分参数
    SR_SCALE = 2  # 超分放大倍数（2倍/3倍/4倍，需匹配预训练模型）
    SR_MODEL_PATH = {  # 超分模型路径（OpenCV预训练模型）
        2: "EDSR_x2.pb",
        3: "EDSR_x3.pb",
        4: "EDSR_x4.pb"
    }
    OCCLUSION_RATIO = (0.05, 0.2)  # 目标遮挡比例（降低）
    WEATHER_TYPES = ["rain", "fog", "snow"]  # 恶劣天气类型
    WEATHER_PROB = 0.3  # 恶劣天气模拟概率（新增，减少极端天气出现）


# 加载超分模型（全局初始化，避免重复加载）
def load_super_resolution_model(scale=2):
    """加载OpenCV超分模型（EDSR）"""
    model_path = Config.SR_MODEL_PATH.get(scale)
    # print(model_path)
    # 检查模型文件是否存在，若不存在则使用默认的ESPCN模型（OpenCV内置）
    if not os.path.exists(model_path):
        print(f"超分模型文件{model_path}不存在，使用OpenCV内置ESPCN模型")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(cv2.samples.findFile(f"ESPCN_x{scale}.pb"))
        sr.setModel("espcn", scale)
    else:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(r"D:\learningJournal\Detection\ultralytics\ultralytics\xpsUtils\utils" + "\\" +model_path)
        sr.setModel("edsr", scale)
    return sr


# 初始化超分模型
sr_model = load_super_resolution_model(Config.SR_SCALE)


# 创建文件夹
def create_dirs():
    for dir_path in [
        Config.OUTPUT_IMG_DIR,
        Config.OUTPUT_LABEL_DIR,
        Config.VISUAL_DIR
    ]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


# 读取YOLO格式标签
def read_yolo_labels(label_path):
    """
    读取YOLO格式的标签文件
    返回：list of [class_id, x_center, y_center, width, height]
    """
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = list(map(float, line.split()))
                    # 验证标签格式是否正确（至少5个值，坐标在0-1之间）
                    if len(parts) >= 5 and all(0 <= p <= 1 for p in parts[1:5]):
                        labels.append(parts)
                except ValueError:
                    continue
    return labels


# 保存YOLO格式标签
def save_yolo_labels(label_path, labels):
    """保存YOLO格式的标签文件"""
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(f"{' '.join(map(str, label))}\n")


# ---------------------- 基础数据增强函数（带操作记录） ----------------------
def random_scale(image, labels, scale_range, aug_ops):
    """随机缩放（记录操作）"""
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h))
    aug_ops.append("scale")  # 记录操作

    # 更新标签（YOLO格式是相对坐标，缩放后不变）
    return image, labels, aug_ops


def random_crop(image, labels, crop_ratio_range, aug_ops):
    """随机比例裁剪（基于原图宽高的比例，记录操作）"""
    h, w = image.shape[:2]
    # 随机选择裁剪比例（宽和高使用相同比例，保持原图比例）
    crop_ratio = random.uniform(*crop_ratio_range)
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    # 确保裁剪尺寸至少为1x1
    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    if h < crop_h or w < crop_w:
        # 尺寸不足时填充（理论上不会出现，因为比例小于1）
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        h, w = image.shape[:2]

    # 随机选择裁剪区域
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    cropped_image = image[y1:y2, x1:x2]
    aug_ops.append(f"crop_{crop_ratio:.2f}")  # 记录裁剪比例

    # 更新标签
    new_labels = []
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        # 转换为绝对坐标
        x_abs = x * w
        y_abs = y * h
        w_abs = w_obj * w
        h_abs = h_obj * h

        # 计算目标在裁剪后的坐标
        x_abs_new = x_abs - x1
        y_abs_new = y_abs - y1

        # 检查目标是否在裁剪区域内（中心在裁剪区域内则保留）
        if 0 <= x_abs_new < crop_w and 0 <= y_abs_new < crop_h:
            x_new = x_abs_new / crop_w
            y_new = y_abs_new / crop_h
            w_new = w_abs / crop_w
            h_new = h_abs / crop_h
            new_labels.append([cls, x_new, y_new, w_new, h_new])

    return cropped_image, new_labels, aug_ops


def random_flip(image, labels, flip_prob=0.5, aug_ops=None):
    """随机水平翻转（记录操作）"""
    if aug_ops is None:
        aug_ops = []
    if random.random() < flip_prob:
        image = cv2.flip(image, 1)
        # 更新标签
        for label in labels:
            label[1] = 1 - label[1]  # x_center = 1 - x_center
        aug_ops.append("flip")  # 记录操作
    return image, labels, aug_ops


def random_rotate(image, labels, rotate_range, aug_ops):
    """随机旋转（记录操作）"""
    h, w = image.shape[:2]
    angle = random.uniform(*rotate_range)
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 旋转图片
    rotated_image = cv2.warpAffine(image, M, (w, h))
    aug_ops.append(f"rotate_{angle:.1f}")  # 记录旋转角度

    # 更新标签
    new_labels = []
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        # 转换为绝对坐标（中心坐标）
        x_abs = x * w
        y_abs = y * h

        # 旋转点坐标
        cos_theta = np.cos(np.radians(-angle))
        sin_theta = np.sin(np.radians(-angle))
        x_rot = (x_abs - w / 2) * cos_theta - (y_abs - h / 2) * sin_theta + w / 2
        y_rot = (x_abs - w / 2) * sin_theta + (y_abs - h / 2) * cos_theta + h / 2

        # 检查是否在图片范围内
        if 0 <= x_rot < w and 0 <= y_rot < h:
            x_new = x_rot / w
            y_new = y_rot / h
            new_labels.append([cls, x_new, y_new, w_obj, h_obj])

    return rotated_image, new_labels, aug_ops


def random_translate(image, labels, translate_range, aug_ops):
    """随机平移（记录操作）"""
    h, w = image.shape[:2]
    tx = random.uniform(*translate_range) * w
    ty = random.uniform(*translate_range) * h

    # 平移矩阵
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (w, h))
    aug_ops.append("translate")  # 记录操作

    # 更新标签
    new_labels = []
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        x_new = x + tx / w
        y_new = y + ty / h

        # 检查中心是否在图片范围内
        if 0 <= x_new <= 1 and 0 <= y_new <= 1:
            new_labels.append([cls, x_new, y_new, w_obj, h_obj])

    return translated_image, new_labels, aug_ops


def adjust_brightness_contrast(image, brightness_range, contrast_range, aug_ops):
    """调整亮度和对比度（记录操作）"""
    # 转换为PIL Image方便处理
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 调整亮度
    brightness = random.uniform(*brightness_range)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)

    # 调整对比度
    contrast = random.uniform(*contrast_range)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)

    # 转换回OpenCV格式
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    aug_ops.append("brightness_contrast")  # 记录操作
    return image, aug_ops


def add_noise(image, noise_type='gaussian', aug_ops=None):
    """添加高斯噪声或椒盐噪声（优化噪声强度，记录操作）"""
    if aug_ops is None:
        aug_ops = []
    if noise_type == 'gaussian':
        # 降低高斯噪声强度，且使用更平缓的噪声分布
        sigma = random.uniform(*Config.GAUSSIAN_NOISE_SIGMA)
        # 对噪声进行归一化，避免过度曝光
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        aug_ops.append("gaussian_noise")  # 记录操作
    elif noise_type == 'salt_pepper':
        # 降低椒盐噪声比例，且只在部分区域添加
        h, w = image.shape[:2]
        # 生成掩码，减少噪点数量
        mask = np.random.rand(h, w) < Config.SALT_PEPPER_NOISE_RATIO
        salt = mask & (np.random.rand(h, w) < 0.5)
        pepper = mask & (np.random.rand(h, w) >= 0.5)

        noisy_image = image.copy()
        noisy_image[salt] = 255
        noisy_image[pepper] = 0
        aug_ops.append("salt_pepper_noise")  # 记录操作
    else:
        noisy_image = image
    return noisy_image, aug_ops


# ---------------------- 小目标增强函数（带操作记录） ----------------------
def is_small_object(label, image_size):
    """判断是否为小目标"""
    h, w = image_size
    w_obj = label[3] * w
    h_obj = label[4] * h
    area = w_obj * h_obj
    total_area = w * h
    return area / total_area < Config.SMALL_OBJECT_THRESH


def super_resolve_object(small_obj_pixel):
    """对小目标像素块进行超分处理"""
    global sr_model
    # 超分放大
    sr_obj = sr_model.upsample(small_obj_pixel)
    return sr_obj


def small_object_oversampling(image, labels, aug_ops):
    """小目标过采样增强（超分+像素与标签缩放同步）"""
    h, w = image.shape[:2]
    # 筛选出所有小目标（带有效像素区域）
    small_labels_with_pixel = []
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        x1_abs = int((x - w_obj / 2) * w)
        y1_abs = int((y - h_obj / 2) * h)
        x2_abs = int((x + w_obj / 2) * w)
        y2_abs = int((y + h_obj / 2) * h)
        if x1_abs >= 0 and y1_abs >= 0 and x2_abs <= w and y2_abs <= h and (x2_abs - x1_abs) > 0 and (
                y2_abs - y1_abs) > 0:
            small_labels_with_pixel.append((label, (x1_abs, y1_abs, x2_abs, y2_abs)))

    if not small_labels_with_pixel:
        return image, labels, aug_ops

    new_image = image.copy()
    new_labels = labels.copy()

    for _ in range(Config.OVERSAMPLING_FACTOR):
        # 随机选一个小目标
        (label, (x1_abs, y1_abs, x2_abs, y2_abs)) = random.choice(small_labels_with_pixel)
        cls, x, y, w_obj, h_obj = label

        # 提取小目标的局部区域
        small_obj_pixel = new_image[y1_abs:y2_abs, x1_abs:x2_abs]
        obj_h, obj_w = small_obj_pixel.shape[:2]

        # 1. 对小目标进行超分处理
        sr_obj = super_resolve_object(small_obj_pixel)
        sr_h, sr_w = sr_obj.shape[:2]

        # 2. 可选：再进行随机缩放（基于超分后的尺寸，等比例）
        scale = random.uniform(1.0, 1.5)  # 超分后轻微缩放
        new_w = int(sr_w * scale)
        new_h = int(sr_h * scale)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        small_obj_scaled = cv2.resize(sr_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. 随机粘贴回原图片（避免越界）
        pos_x = random.randint(0, max(0, w - new_w))
        pos_y = random.randint(0, max(0, h - new_h))
        new_image[pos_y:pos_y + new_h, pos_x:pos_x + new_w] = small_obj_scaled

        # 4. 计算新标签（严格基于超分+缩放后的像素尺寸，等比例）
        x_new = (pos_x + new_w / 2) / w
        y_new = (pos_y + new_h / 2) / h
        w_new = new_w / w  # 宽占比同步缩放
        h_new = new_h / h  # 高占比同步缩放
        new_labels.append([cls, x_new, y_new, w_new, h_new])

    aug_ops.append("small_oversampling_sr")  # 记录超分操作
    return new_image, new_labels, aug_ops


def synthesize_dense_small_objects(image, labels, aug_ops):
    """密集小目标合成增强（超分+像素+标签同步缩放，避免比例失调）"""
    h, w = image.shape[:2]
    # 筛选出所有小目标（带有效像素区域）
    small_labels_with_pixel = []
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        x1_abs = int((x - w_obj / 2) * w)
        y1_abs = int((y - h_obj / 2) * h)
        x2_abs = int((x + w_obj / 2) * w)
        y2_abs = int((y + h_obj / 2) * h)
        if x1_abs >= 0 and y1_abs >= 0 and x2_abs <= w and y2_abs <= h and (x2_abs - x1_abs) > 0 and (
                y2_abs - y1_abs) > 0:
            small_labels_with_pixel.append((label, (x1_abs, y1_abs, x2_abs, y2_abs)))

    if not small_labels_with_pixel:
        return image, labels, aug_ops

    new_image = image.copy()
    new_labels = labels.copy()

    # 按合成数量循环生成新小目标
    for _ in range(Config.SYNTHESIS_NUM):
        # 随机选一个现有小目标（含像素区域）
        (label, (x1_abs, y1_abs, x2_abs, y2_abs)) = random.choice(small_labels_with_pixel)
        cls, _, _, w_obj, h_obj = label

        # 1. 裁剪小目标的像素块
        small_obj_pixel = new_image[y1_abs:y2_abs, x1_abs:x2_abs]
        obj_h, obj_w = small_obj_pixel.shape[:2]

        # 2. 对小目标进行超分处理
        sr_obj = super_resolve_object(small_obj_pixel)
        sr_h, sr_w = sr_obj.shape[:2]

        # 3. 随机缩放小目标（基于超分后的尺寸，等比例）
        synth_scale = random.uniform(*Config.SYNTHESIS_SCALE_RANGE)
        new_w = int(sr_w * synth_scale)
        new_h = int(sr_h * synth_scale)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        small_obj_scaled = cv2.resize(sr_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 4. 随机选择新位置（避免贴到图片边缘）
        pos_x = random.randint(0, max(0, w - new_w))
        pos_y = random.randint(0, max(0, h - new_h))

        # 5. 将小目标像素块粘贴到新位置
        new_image[pos_y:pos_y + new_h, pos_x:pos_x + new_w] = small_obj_scaled

        # 6. 计算新标签（严格基于超分+缩放后的像素尺寸，等比例）
        x_new = (pos_x + new_w / 2) / w
        y_new = (pos_y + new_h / 2) / h
        w_new = new_w / w  # 宽占比同步缩放
        h_new = new_h / h  # 高占比同步缩放
        new_labels.append([cls, x_new, y_new, w_new, h_new])

    aug_ops.append("small_synthesis_sr")  # 记录超分操作
    return new_image, new_labels, aug_ops


def enhance_small_object_features(image, aug_ops):
    """小目标特征增强（锐化，降低锐化强度，记录操作）"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 降低锐化强度（从2.0改为1.5）
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)
    # 轻微边缘增强
    pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE if random.random() < 0.3 else ImageFilter.EDGE_ENHANCE)
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    aug_ops.append("small_feature_enhance")  # 记录操作
    return image, aug_ops


def background_disturbance(image, aug_ops):
    """背景扰动增强（降低扰动强度，记录操作）"""
    h, w = image.shape[:2]
    # 降低噪声强度，且只在50%的区域添加扰动
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.uint8)
    mask = np.random.rand(h, w) > 0.8  # 20%的区域添加扰动（原70%）
    image[mask] = cv2.addWeighted(image[mask], 0.8, noise[mask], 0.2, 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    aug_ops.append("background_disturb")  # 记录操作
    return image, aug_ops


def object_occlusion(image, labels, aug_ops):
    """目标部分遮挡（修复空切片和NoneType错误，记录操作）"""
    h, w = image.shape[:2]
    new_image = image.copy()

    for label in labels:
        cls, x, y, w_obj, h_obj = label
        # 转换为绝对坐标
        x_abs = x * w
        y_abs = y * h
        w_abs = w_obj * w
        h_abs = h_obj * h

        # 跳过无效的目标（宽高为0）
        if w_abs <= 0 or h_abs <= 0:
            continue

        # 计算遮挡区域（添加有效性检查）
        occlusion_w = int(w_abs * random.uniform(*Config.OCCLUSION_RATIO))
        occlusion_h = int(h_abs * random.uniform(*Config.OCCLUSION_RATIO))
        # 避免遮挡区域为0
        if occlusion_w <= 0 or occlusion_h <= 0:
            continue

        x1 = int(x_abs - occlusion_w / 2)
        y1 = int(y_abs - occlusion_h / 2)
        x2 = x1 + occlusion_w
        y2 = y1 + occlusion_h

        # 避免越界（强制修正坐标）
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 跳过空切片（x1 >= x2 或 y1 >= y2）
        if x1 >= x2 or y1 >= y2:
            continue

        # 随机遮挡（使用半透明遮挡，更自然）
        occlusion_type = random.choice(['black', 'white', 'noise', 'translucent'])
        if occlusion_type == 'black':
            # 黑色半透明遮挡
            mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            new_image[y1:y2, x1:x2] = cv2.addWeighted(new_image[y1:y2, x1:x2], 0.7, mask, 0.3, 0)
        elif occlusion_type == 'white':
            # 白色半透明遮挡
            mask = np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * 255
            new_image[y1:y2, x1:x2] = cv2.addWeighted(new_image[y1:y2, x1:x2], 0.7, mask, 0.3, 0)
        elif occlusion_type == 'noise':
            # 噪声遮挡（确保噪声维度匹配）
            noise = np.random.randint(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            new_image[y1:y2, x1:x2] = cv2.addWeighted(new_image[y1:y2, x1:x2], 0.8, noise, 0.2, 0)
        else:
            # 半透明灰色遮挡
            mask = np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * 128
            new_image[y1:y2, x1:x2] = cv2.addWeighted(new_image[y1:y2, x1:x2], 0.9, mask, 0.1, 0)

    aug_ops.append("object_occlusion")  # 记录操作
    return new_image, aug_ops


def simulate_bad_weather(image, weather_type, aug_ops):
    """模拟恶劣天气（降低天气强度，记录操作）"""
    h, w = image.shape[:2]
    if weather_type == 'rain':
        # 减少雨丝数量，降低亮度
        rain = np.random.rand(h, w) > 0.995  # 原0.98
        image[rain] = cv2.add(image[rain], np.array([50], dtype=np.uint8))
        aug_ops.append(f"weather_rain")  # 记录操作
    elif weather_type == 'fog':
        # 降低雾的浓度
        fog = np.ones_like(image) * random.randint(180, 220)  # 原150-200
        image = cv2.addWeighted(image, 0.85, fog, 0.15, 0)  # 原0.7和0.3
        aug_ops.append(f"weather_fog")  # 记录操作
    elif weather_type == 'snow':
        # 减少雪花数量
        snow = np.random.rand(h, w) > 0.997  # 原0.99
        image[snow] = 255
        aug_ops.append(f"weather_snow")  # 记录操作
    return image, aug_ops


# ---------------------- 主增强流程（带操作记录） ----------------------
def augment_image(image_path, label_path):
    """对单张图片进行增强（记录所有操作）"""
    # 初始化操作记录列表
    aug_ops = []

    # 读取图片和标签
    image = cv2.imread(image_path)
    if image is None:
        return None, None, aug_ops
    labels = read_yolo_labels(label_path)

    # 基础增强
    image, labels, aug_ops = random_scale(image, labels, Config.SCALE_RANGE, aug_ops)
    image, labels, aug_ops = random_crop(image, labels, Config.CROP_RATIO_RANGE, aug_ops)
    image, labels, aug_ops = random_flip(image, labels, aug_ops=aug_ops)
    image, labels, aug_ops = random_rotate(image, labels, Config.ROTATE_RANGE, aug_ops)
    image, labels, aug_ops = random_translate(image, labels, Config.TRANSLATE_RANGE, aug_ops)
    image, aug_ops = adjust_brightness_contrast(image, Config.BRIGHTNESS_RANGE, Config.CONTRAST_RANGE, aug_ops)

    # 添加噪声（降低概率，且随机选择是否添加）
    if random.random() < Config.NOISE_PROB:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'none'])  # 增加none选项
        if noise_type != 'none':
            image, aug_ops = add_noise(image, noise_type, aug_ops)

    # 小目标增强（含超分）
    image, labels, aug_ops = small_object_oversampling(image, labels, aug_ops)
    image, labels, aug_ops = synthesize_dense_small_objects(image, labels, aug_ops)
    image, aug_ops = enhance_small_object_features(image, aug_ops)
    image, aug_ops = background_disturbance(image, aug_ops)
    image, aug_ops = object_occlusion(image, labels, aug_ops)

    # 恶劣天气模拟（降低概率）
    if random.random() < Config.WEATHER_PROB:
        weather_type = random.choice(Config.WEATHER_TYPES)
        image, aug_ops = simulate_bad_weather(image, weather_type, aug_ops)

    # 去重并返回操作列表（避免重复记录）
    aug_ops = list(dict.fromkeys(aug_ops))
    return image, labels, aug_ops


# ---------------------- 可视化函数（补充黑边而非拉伸） ----------------------
def draw_bboxes(image, labels):
    """在图片上绘制检测框"""
    h, w = image.shape[:2]
    for label in labels:
        cls, x, y, w_obj, h_obj = label
        # 转换为绝对坐标
        x1 = int((x - w_obj / 2) * w)
        y1 = int((y - h_obj / 2) * h)
        x2 = int((x + w_obj / 2) * w)
        y2 = int((y + h_obj / 2) * h)
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制类别
        cv2.putText(image, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def pad_image_with_black(image, target_size):
    """将图片补充黑边至目标尺寸，保持原比例"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # 计算填充的边距
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # 补充黑边
    padded_image = cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded_image


def visualize_comparison(original_image, original_labels, augmented_image, augmented_labels, save_path):
    """可视化对比原图和增强图（补充黑边而非拉伸）"""
    # 绘制检测框
    original_img = draw_bboxes(original_image.copy(), original_labels)
    augmented_img = draw_bboxes(augmented_image.copy(), augmented_labels)

    # 获取两张图的尺寸
    h1, w1 = original_img.shape[:2]
    h2, w2 = augmented_img.shape[:2]

    # 确定目标尺寸（取最大宽和最大高）
    target_w = max(w1, w2)
    target_h = max(h1, h2)

    # 对两张图分别补充黑边至目标尺寸
    original_img_padded = pad_image_with_black(original_img, (target_h, target_w))
    augmented_img_padded = pad_image_with_black(augmented_img, (target_h, target_w))

    # 拼接图片
    combined = np.hstack((original_img_padded, augmented_img_padded))

    # 添加标题
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(combined, "Augmented", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 保存图片
    cv2.imwrite(save_path, combined)


# ---------------------- 主函数 ----------------------
def main():
    # 创建文件夹
    create_dirs()

    # 遍历所有图片
    for img_name in os.listdir(Config.INPUT_IMG_DIR):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        # 构建路径
        img_path = os.path.join(Config.INPUT_IMG_DIR, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(Config.INPUT_LABEL_DIR, label_name)

        # 读取原图和标签（用于可视化）
        original_image = cv2.imread(img_path)
        original_labels = read_yolo_labels(label_path)

        # 数据增强（获取操作记录）
        augmented_image, augmented_labels, aug_ops = augment_image(img_path, label_path)
        if augmented_image is None:
            print(f"跳过无效图片：{img_name}")
            continue

        # 生成带操作标记的文件名
        img_base, img_ext = os.path.splitext(img_name)
        label_base = os.path.splitext(label_name)[0]
        # 将操作列表转换为字符串（用下划线连接）
        ops_str = "_".join(aug_ops) if aug_ops else "no_aug"
        # 新文件名：原名称_操作列表.后缀
        new_img_name = f"{img_base}_{ops_str}{img_ext}"
        new_label_name = f"{label_base}_{ops_str}.txt"

        # 保存增强后的图片和标签
        output_img_path = os.path.join(Config.OUTPUT_IMG_DIR, new_img_name)
        output_label_path = os.path.join(Config.OUTPUT_LABEL_DIR, new_label_name)
        cv2.imwrite(output_img_path, augmented_image)
        save_yolo_labels(output_label_path, augmented_labels)

        # 可视化对比（补充黑边）
        vis_name = f"compare_{img_base}_{ops_str}{img_ext}"
        vis_path = os.path.join(Config.VISUAL_DIR, vis_name)
        visualize_comparison(original_image, original_labels, augmented_image, augmented_labels, vis_path)

        print(f"处理完成：{img_name} -> {new_img_name}")


if __name__ == "__main__":
    main()