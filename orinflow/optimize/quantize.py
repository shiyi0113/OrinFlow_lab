"""Quantization utilities (PTQ and QAT)."""

import re
from pathlib import Path
from typing import List

import numpy as np
import onnx

from orinflow.config import ONNX_OPTIMIZED_DIR, ONNX_RAW_DIR, DEFAULT_INPUT_SIZE, DEFAULT_CALIB_SAMPLES, print_trtexec_hint
from orinflow.data.calibration import get_calibration_data

def _get_calibration_device() -> str:
    """
    Auto-detect the best device for calibration.

    Returns:
        Device string ("cuda:0" or "cpu").
    """
    try:
        from orinflow.runtime import check_onnxruntime

        info = check_onnxruntime()
        if info.has_cuda:
            return "cuda:0"
    except ImportError:
        pass

    # Fallback: check torch CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass

    return "cpu"

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
    data_yaml: str,
    num_calib_images: int = DEFAULT_CALIB_SAMPLES,
    imgsz: int = DEFAULT_INPUT_SIZE,
    quantize_mode: str = "int8",
    calibration_method: str = "entropy",
    nodes_to_exclude: list[str] | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
) -> Path:
    """
    Quantize ONNX model using ModelOpt PTQ.
    """
    from modelopt.onnx.quantization import quantize

    if input_dir is None:
        input_dir = ONNX_RAW_DIR
    if output_dir is None:
        output_dir = ONNX_OPTIMIZED_DIR

    model_path = input_dir / model_name
    output_path = output_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    print(f"Generating calibration data from {data_yaml}...")
    calib_data = get_calibration_data(
        yaml_file=data_yaml, 
        split="train", # 默认为 train，也可暴露参数
        num_images=num_calib_images, 
        input_size=imgsz
    )
    print(f"Calibration data shape: {calib_data.shape}")

    # device
    real_nodes_to_exclude = _resolve_exclusion_patterns(model_path, nodes_to_exclude)
    if device is None:
        device = _get_calibration_device()
    calibration_eps = [device]
    print(f"Using calibration device: {device}")

    # Quantize
    print(f"Quantizing {model_path} with {quantize_mode} mode...")
    quantize(
        onnx_path=str(model_path),
        output_path=str(output_path),
        calibration_data=calib_data,
        calibration_method=calibration_method,
        quantize_mode=quantize_mode,
        nodes_to_exclude=real_nodes_to_exclude,
        calibration_eps=calibration_eps,
        calibration_shapes=f"images:1x3x{imgsz}x{imgsz}",
    )

    print(f"Quantization complete: {output_path}")
    sparse = "sparse" in output_path.stem.lower()
    print_trtexec_hint(output_path, sparse=sparse)
    return output_path