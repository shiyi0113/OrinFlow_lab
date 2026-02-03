"""Quantization utilities (PTQ and QAT)."""

from pathlib import Path
import numpy as np
import onnx
import re
from typing import List
from orinflow.config import CALIB_DIR, ONNX_OPTIMIZED_DIR, ONNX_RAW_DIR, print_trtexec_hint

def _resolve_exclusion_patterns(model_path: Path, patterns: List[str] | None) -> List[str] | None:
    """
    解析排除列表：支持正则表达式和模糊匹配。
    """
    if not patterns:
        return None

    print(f"Resolving exclusion patterns: {patterns}")
    
    # 加载 ONNX 模型以获取所有节点名称
    model = onnx.load(str(model_path), load_external_data=False)
    all_node_names = [node.name for node in model.graph.node]
    
    resolved_nodes = set()
    
    for pattern in patterns:
        # 创建正则对象，忽略大小写
        # 如果输入的是普通字符串，re.search 也会尝试匹配它
        regex = re.compile(pattern, re.IGNORECASE)
        
        matched_count = 0
        for name in all_node_names:
            if regex.search(name):
                resolved_nodes.add(name)
                matched_count += 1
        
        if matched_count == 0:
            print(f"⚠️ Warning: Pattern '{pattern}' did not match any nodes in the model.")
        else:
            print(f"✅ Pattern '{pattern}' matched {matched_count} nodes.")

    final_list = list(resolved_nodes)
    if final_list:
        print(f"🚫 Total {len(final_list)} nodes will be EXCLUDED from quantization.")
        # 打印前5个作为示例
        print(f"   Example excluded nodes: {final_list[:5]} ...")
        
    return final_list

def quantize_onnx(
    model_name: str,
    calib_name: str,
    quantize_mode: str = "int8",
    calibration_method: str = "entropy",
    nodes_to_exclude: list[str] | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Quantize ONNX model using ModelOpt PTQ.

    Args:
        model_name: Model name (without .onnx extension)
        calib_name: Calibration data name (without _calib.npy extension)
        quantize_mode: Quantization mode (int8, int4, fp8)
        calibration_method: Calibration method (entropy, max, awq_clip, etc.)
        nodes_to_exclude: List of node name patterns to exclude from quantization
        input_dir: Input directory, defaults to models/onnx_raw
        output_dir: Output directory, defaults to models/onnx_quant

    Returns:
        Path to quantized ONNX model
    """
    from modelopt.onnx.quantization import quantize

    if input_dir is None:
        input_dir = ONNX_RAW_DIR
    if output_dir is None:
        output_dir = ONNX_OPTIMIZED_DIR

    model_path = input_dir / model_name
    calib_path = CALIB_DIR / calib_name
    output_path = output_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {calib_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    calib_data = np.load(calib_path)
    print(f"Calibration data shape: {calib_data.shape}")
    real_nodes_to_exclude = _resolve_exclusion_patterns(model_path, nodes_to_exclude)

    # Quantize
    print(f"Quantizing {model_path} with {quantize_mode} mode...")
    quantize(
        onnx_path=str(model_path),
        output_path=str(output_path),
        calibration_data=calib_data,
        calibration_method=calibration_method,
        quantize_mode=quantize_mode,
        nodes_to_exclude=real_nodes_to_exclude,
        calibration_eps=["cuda:0"],
        calibration_shapes="images:1x3x640x640",
    )

    print(f"Quantization complete: {output_path}")
    sparse = "sparse" in output_path.stem.lower()
    print_trtexec_hint(output_path, sparse=sparse)
    return output_path