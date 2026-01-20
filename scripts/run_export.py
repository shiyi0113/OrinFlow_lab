from ultralytics import YOLO
import shutil
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录
PROJECT_ROOT = BASE_DIR.parent              # 项目目录

model_folder = PROJECT_ROOT / "models" / "pytorch"
output_folder = PROJECT_ROOT / "models" / "onnx_raw"
model_name = 'yolo26l.pt'
output_name = 'yolo26l.onnx'

model_path = os.path.join(model_folder, model_name)
output_path = os.path.join(output_folder,output_name)

print(f"正在加载模型: {model_path} ...")
model = YOLO(model_path)

if os.path.exists(model_name):
    shutil.move(model_name, model_path)
    print(f"模型下载完成并移动至: {model_path}")
else:
    print(f"模型已下载，但不在当前目录。请检查你的文件系统。")

print("开始导出 ONNX ...")
path = model.export(
    format='onnx',   
    opset=17,
    half=False,         
    simplify=True,    
    dynamic=False,    
    imgsz=640,
    device=0,     
)

if os.path.exists(path):
    shutil.move(path, output_path)
    print(f"模型转换完成并移动至: {output_path}")
else:
    print(f"模型已转换，但不在当前目录。请检查你的文件系统。")