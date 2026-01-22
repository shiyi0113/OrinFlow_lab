import argparse
import numpy as np
from pathlib import Path
from modelopt.onnx.quantization import quantize

BASE_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = BASE_DIR.parent     

def main():
    parser = argparse.ArgumentParser(description="ONNX 模型PTQ量化")
    parser.add_argument("-m", "--model",type=str,default="yolo26l.onnx",help="ONNX 模型文件名 (默认为 yolo26l.onnx)")
    parser.add_argument("--calib",type=str,default="coco128",help="校验数据集名称") 
    args = parser.parse_args()
    model_path = PROJECT_ROOT / "models" / "onnx_raw" / args.model
    output_path = PROJECT_ROOT / "models" / "onnx_quant" / args.model   
    calib_path = PROJECT_ROOT / "data" / args.calib / "yolo_calib.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"找不到校准数据: {calib_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        calib_data = np.load(calib_path)
        print(f"校准数据形状: {calib_data.shape}")
        quantize(
            onnx_path=str(model_path),
            output_path=str(output_path),
            calibration_data=calib_data,
            calibration_method="entropy",
            quantize_mode="int8",
        )
        print(f"3. 量化完成！输出文件: {output_path}")
    except Exception as e:
        print(f"量化过程中发生错误: {e}")

if __name__ == "__main__":
    main()