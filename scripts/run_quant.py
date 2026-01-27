import onnx
import argparse
import numpy as np
from pathlib import Path
from modelopt.onnx.quantization import quantize

BASE_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = BASE_DIR.parent     

def main():
    parser = argparse.ArgumentParser(description="ONNX 模型PTQ量化")
    parser.add_argument("-m", "--model",type=str,default="yolo26l",help="ONNX 模型文件名 (默认为 yolo26l)")
    parser.add_argument("--calib",type=str,default="coco128",help="校验数据集名称 (默认为 coco128)") 
    args = parser.parse_args()
    model_path = PROJECT_ROOT / "models" / "onnx_raw" / f"{args.model}.onnx"
    output_path = PROJECT_ROOT / "models" / "onnx_quant" / f"{args.model}.onnx"  
    calib_path = PROJECT_ROOT / "data" / "calib" / f"{args.calib}_calib.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"找不到校准数据: {calib_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 节点排除
    # model = onnx.load(model_path)
    # exclude_nodes = []
    # for node in model.graph.node:
    #     # if "model.23" in node.name:             #  Detect
    #     #     exclude_nodes.append(node.name)
    #     # if "model.10" in node.name:             #  C2PSA
    #     #     exclude_nodes.append(node.name)
    #     if '/model.10/' in node.name and '/attn/' in node.name:
    #         exclude_nodes.append(node.name)
    #     if node.op_type in ['Reshape', 'Shape', 'Gather', 'Concat', 'Unsqueeze']:
    #         exclude_nodes.append(node.name)
    # print(f"排除节点数: {len(exclude_nodes)}")

    try:
        calib_data = np.load(calib_path)
        print(f"校准数据形状: {calib_data.shape}")
        quantize(
            onnx_path=str(model_path),
            output_path=str(output_path),
            calibration_data=calib_data,
            # nodes_to_exclude=exclude_nodes,
            calibration_method="entropy",
            quantize_mode="int8",
            calibration_eps=["cuda:0"],
            calibration_shapes="images:1x3x640x640"
        )
        print(f"量化完成！输出文件: {output_path}")
    except Exception as e:
        print(f"量化过程中发生错误: {e}")

if __name__ == "__main__":
    main()