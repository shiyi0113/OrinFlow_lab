import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# 路径定义
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

def main():
    parser = argparse.ArgumentParser(description="模型精度对齐验证 (PyTorch vs ONNX Int8)")
    parser.add_argument("-m", "--model", type=str, default="yolo26l", help="模型名称 (不含后缀)")
    parser.add_argument("--yaml", type=str, default="coco128.yaml", help="数据集配置文件")
    args = parser.parse_args()

    # 1. 定义文件路径
    pt_path = PROJECT_ROOT / "models" / "pytorch" / f"{args.model}.pt"
    onnx_quant_path = PROJECT_ROOT / "models" / "onnx_quant" / f"{args.model}.onnx"
    data_yaml_path = PROJECT_ROOT / "data" / "images" / args.yaml

    # 检查文件
    if not pt_path.exists():
        print(f"❌ 错误: 找不到 PyTorch 模型: {pt_path}")
        return
    if not onnx_quant_path.exists():
        print(f"❌ 错误: 找不到量化模型: {onnx_quant_path}")
        return
    if not data_yaml_path.exists():
        print(f"❌ 错误: 找不到数据集配置: {data_yaml_path}")
        return

    results = []

    # ---------------------------------------------------------
    # 2. 评估 PyTorch 模型 (基准)
    # ---------------------------------------------------------
    print(f"\n🚀 开始评估 PyTorch 基准模型: {pt_path.name} ...")
    try:
        model_pt = YOLO(pt_path)
        metrics_pt = model_pt.val(data=str(data_yaml_path), verbose=False, workers=0)
        
        pt_map50 = metrics_pt.box.map50
        pt_map = metrics_pt.box.map
        results.append({"Format": "PyTorch (FP32)", "mAP50": pt_map50, "mAP50-95": pt_map})
        print(f"✅ PyTorch 验证完成: mAP50={pt_map50:.4f}, mAP50-95={pt_map:.4f}")
    except Exception as e:
        print(f"❌ PyTorch 验证失败: {e}")
        return

    # ---------------------------------------------------------
    # 3. 评估 ONNX Quant 模型
    # ---------------------------------------------------------
    print(f"\n🚀 开始评估 ONNX Int8 模型: {onnx_quant_path.name} ...")
    print("   (这可能需要几分钟，Ultralytics 会自动调用 ONNX Runtime)")
    try:
        model_onnx = YOLO(onnx_quant_path)
        metrics_onnx = model_onnx.val(data=str(data_yaml_path), task='detect', verbose=False, workers=0)
        
        onnx_map50 = metrics_onnx.box.map50
        onnx_map = metrics_onnx.box.map
        results.append({"Format": "ONNX (Int8)", "mAP50": onnx_map50, "mAP50-95": onnx_map})
        print(f"✅ ONNX 验证完成: mAP50={onnx_map50:.4f}, mAP50-95={onnx_map:.4f}")
    except Exception as e:
        print(f"❌ ONNX 验证失败: {e}")
        print("💡 提示: 请确保当前环境中安装了 onnxruntime-gpu (pip install onnxruntime-gpu)")
        return

    # ---------------------------------------------------------
    # 4. 输出对比报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 精度对齐报告")
    print("="*50)
    df = pd.DataFrame(results)
    
    if len(df) == 2:
        diff_map = pt_map - onnx_map
        drop_rate = (diff_map / pt_map) * 100
        print(df.to_string(index=False))
        print("-" * 50)
        print(f"📉 mAP50-95 绝对下降: {diff_map:.4f}")
        print(f"📉 精度损失率: {drop_rate:.2f}%")
        
        if drop_rate < 1.0:
            print("✅ 结论: 精度损失极小，量化非常成功！")
        elif drop_rate < 5.0:
            print("⚠️ 结论: 精度有一定损失，建议检查敏感层或尝试 QAT。")
        else:
            print("❌ 结论: 精度崩塌！请务必进行 Polygraphy 逐层排查。")
    else:
        print(df)

if __name__ == "__main__":
    main()