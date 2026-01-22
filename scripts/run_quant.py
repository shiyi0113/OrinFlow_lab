import os
import numpy as np
from pathlib import Path
from modelopt.onnx.quantization import quantize

BASE_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = BASE_DIR.parent     

model_path = PROJECT_ROOT / "models" / "onnx_raw" / "yolo26l.onnx"
output_path = PROJECT_ROOT / "models" / "onnx_quant" / "yolo26l.onnx"
calib_path = PROJECT_ROOT / "data" / "coco128" / "yolo_calib.npy"

if __name__ == "__main__":

    quantize(
        onnx_path=str(model_path),
        output_path=str(output_path),
        calibration_data=np.load(calib_path), 
        calibration_method="entropy",
        quantize_mode="int8",
    )
    print(f"3. 量化完成！输出文件: {output_path}")