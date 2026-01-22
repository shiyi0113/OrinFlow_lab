import os
import glob
import numpy as np
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = BASE_DIR.parent     

model_path = PROJECT_ROOT / "models" / "onnx_raw" / "yolo26l.onnx"
output_path = PROJECT_ROOT / "models" / "onnx_quant" / "yolo26l.onnx"
images_dir = PROJECT_ROOT / "data" / "coco128" / "images" / "train2017"
calib_path = PROJECT_ROOT / "data" / "coco128" / "yolo_calib.npy"
num_images = 128
input_size = 640

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # img: 输入图片(HWC, RGB)
    # new_shape: 目标尺寸 (w, h)
    # color: 填充色
    # scaleFill: True=直接Resize（对应简化版），False=Letterbox（默认）
    
    shape = img.shape[:2]  # 原始尺寸 (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    if not scaleup:  # 只缩小，不放大（避免低分辨率图片放大模糊）
        r = min(r, 1.0)

    # 计算缩放后的尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # 计算填充量

    if auto:  # 最小填充原则（让填充量为32的倍数，适配YOLO的下采样）
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:  # 直接拉伸（简化版）
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[0], new_shape[1])

    dw /= 2  # 左右均分填充（更合理，而非只填充右侧/底部）
    dh /= 2

    # 缩放图片
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 填充图片
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess_yolo_image(img_path, input_size=(640, 640)):
    """
    读取并预处理图片，使其符合 YOLO 的输入要求。
    注意：这里使用了最简单的 Resize。如果你的推理流程包含 Letterbox (填充黑边)，
    请务必在这里也实现一致的逻辑，否则量化精度会下降。
    """
    # 1. 读取图片 (H, W, C) - BGR 格式
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 2. 颜色转换 BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. 缩放 (Resize) 到模型输入尺寸 (例如 640x640)
    img, _, _ = letterbox(img, new_shape=(640, 640))

    # 4. 归一化 (Normalize) 0-255 -> 0.0-1.0
    img = img.astype(np.float32) / 255.0

    # 5. 维度置换 (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    
    # 6. 增加 Batch 维度 (1, C, H, W) -> 这一步在堆叠时处理，单张图先不加
    return img

def main():
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png"))
    
    image_paths = image_paths[:num_images]
    print(f"找到 {len(image_paths)} 张图片用于校准...")

    calib_data = []
    for p in image_paths:
        img_tensor = preprocess_yolo_image(p, input_size=(input_size, input_size))
        if img_tensor is not None:
            calib_data.append(img_tensor)

    if not calib_data:
        print("错误：未处理任何图片，请检查路径。")
        return

    calib_blob = np.stack(calib_data, axis=0)
    
    print(f"校准数据形状: {calib_blob.shape}")
    print(f"保存校准数据至: {calib_path}")
    np.save(calib_path, calib_blob)

if __name__ == "__main__":
    main()