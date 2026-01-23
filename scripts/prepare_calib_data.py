"""
为 YOLO 模型准备校准数据
"""
import os
import shutil
import numpy as np
from pathlib import Path
import cv2
import yaml
import zipfile
import argparse
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = BASE_DIR.parent     
DATA_ROOT = PROJECT_ROOT / "data" / "images"
CALIB_ROOT = PROJECT_ROOT / "data" / "calib"

def letterbox(
    img, 
    new_shape=(640, 640), 
    color=(114, 114, 114), 
    auto=True, 
    scaleFill=False, 
    scaleup=True,
    stride=32,
):
    """将图片调整为指定大小，保持长宽比（YOLO 标准预处理）。

    Args:
        img: 输入图片 (HWC)
        new_shape: 目标尺寸 (height, width)
        color: 填充颜色
        auto: 最小矩形填充
        scaleFill: 拉伸填充
        scaleup: 是否允许放大
        stride: 步长（用于对齐）
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_yolo_image(image_path, input_size=(640, 640)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Letterbox resize（保持长宽比）
    img, _, _ = letterbox(img, new_shape=input_size,auto=False)
    # 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0
    # HWC to CHW
    img = img.transpose(2, 0, 1)
    return img

def load_image(yaml_file,split='train',num_images=500):
    """根据yaml文件获取图片路径。

    Args:
        yaml_file: yaml文件路径
        split: 使用哪个数据集分割(train, val,test)
        num_images: 要加载的图片数量

    Returns:
        图片路径列表
    """
    print(f"正在读取配置文件: {yaml_file}")
    with open(yaml_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 获取参数
    dataset_dir_name = cfg.get('path')
    calib_dir        = cfg.get(split)
    download_url     = cfg.get('download')

    if not dataset_dir_name:
        raise ValueError(f"错误：配置文件 {yaml_file} 未定义 'path' 字段,未知数据集名称。")
    dataset_root = DATA_ROOT / dataset_dir_name

    if not dataset_root.exists():
        print(f"数据集目录 {dataset_root} 不存在，准备下载...")
        if not download_url:
            raise ValueError(f"错误：配置文件 {yaml_file} 未定义 'download' 字段,无法下载。")
        # 下载
        zip_path = DATA_ROOT / f"{dataset_dir_name}.zip"
        print(f"从 {download_url} 下载...")
        try:
            urlretrieve(download_url,
                        zip_path, 
                        reporthook=lambda count, block_size, total_size: print(
                            f"\r正在下载: {int(count * block_size * 100 / total_size)}%", 
                            end=''
                        )
            )
            print("\n下载完成。")
        except Exception as e:
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"下载失败: {e}")

        # 解压
        print(f"正在解压到 {DATA_ROOT} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_dir = DATA_ROOT / dataset_dir_name
            zip_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(zip_dir)
        
        extracted_items = [
            item for item in zip_dir.iterdir() 
            if not item.name.startswith('.') and not item.name.startswith('__')
        ]
        # 如果目录下只有一个东西，且这个东西是文件夹，说明嵌套了一层
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            nested_folder = extracted_items[0]
            for sub_item in nested_folder.iterdir():
                shutil.move(str(sub_item), str(zip_dir))
            nested_folder.rmdir()
        else:
            os.remove(zip_path)

        if zip_path.exists():
            zip_path.unlink()
        print("解压完成。")
    else:
        print(f"数据集已存在: {dataset_root}")

    if not calib_dir:
        raise ValueError(f"错误：配置文件 {yaml_file} 中未定义 '{split}' 字段，无法定位图片。")
    
    if os.path.isabs(calib_dir):
        images_dir = Path(calib_dir)
    else:
        images_dir = dataset_root / calib_dir
    
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(images_dir.glob(ext)))

    if len(image_files) == 0:
        raise ValueError(f"在 {images_dir} 中没有找到图片")
    
    # 随机采样
    np.random.seed(42) 
    if len(image_files) > num_images:
        image_files = np.random.choice(image_files, num_images, replace=False).tolist()
    print(f"找到 {len(image_files)} 张图片用于校准...")

    return image_files

def main():
    parser = argparse.ArgumentParser(description='量化数据校准集准备脚本')
    parser.add_argument('--yaml_file', type=str, default='coco128.yaml', help='数据集配置文件名')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='选择数据集划分: train, val, 或 test')
    parser.add_argument('--num_images', type=int, default=500, help='用于校准的图片数量')
    parser.add_argument('--input_size', type=int, default=640, help='模型输入尺寸')
    args = parser.parse_args()
    
    YAML_PATH = DATA_ROOT / args.yaml_file
    CALIB_PATH = CALIB_ROOT / f"{Path(args.yaml_file).stem}_calib.npy"

    try:
        image_paths = load_image(YAML_PATH,args.split,args.num_images)
    except Exception as e:
        print(f"错误: 准备数据集时发生问题 - {e}")
        return
    if len(image_paths) == 0:
        print(f"警告: 未找到任何图片！请检查解压是否成功或路径是否正确。")
        return

    # 处理图片
    print(f"\n正在预处理图片...")
    calib_data = []
    for p in image_paths:
        img_tensor = preprocess_yolo_image(p, input_size=(args.input_size, args.input_size))
        if img_tensor is not None:
            calib_data.append(img_tensor)
        else:
            print(f"无法读取图片: {p}")

    if len(calib_data) == 0:
        raise ValueError("没有成功处理任何图片")

    calib_blob = np.stack(calib_data, axis=0)
    print(f"\n校准数据形状: {calib_blob.shape}")
    print(f"数据类型: {calib_blob.dtype}")
    print(f"数值范围: [{calib_blob.min():.3f}, {calib_blob.max():.3f}]")

    np.save(CALIB_PATH, calib_blob)
    print(f"\n校准数据已保存到: {CALIB_PATH}")
    print(f"文件大小: {os.path.getsize(CALIB_PATH) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()