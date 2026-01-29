import onnx_graphsurgeon as gs
import onnx
import numpy as np

# 1. 设置路径
input_model = "models/onnx/optimized/yolo26x_INT8.onnx"
output_model = "yolo26x_INT8_READY_TO_RUN_v3.onnx"

# 2. 定义映射
node_map = {
    "/model.16/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.16/cv2/act/Mul_output_0",
    "/model.19/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.19/cv2/act/Mul_output_0",
    "/model.22/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.22/cv2/act/Mul_output_0"
}

print(f"加载模型: {input_model}...")
graph = gs.import_onnx(onnx.load(input_model))

# 3. 防止名字冲突
for tensor in graph.tensors().values():
    if tensor.name in node_map.values():
        tensor.name = tensor.name + "_original_internal"

count = 0
for int8_name, target_name in node_map.items():
    tensors = [t for t in graph.tensors().values() if t.name == int8_name]
    if not tensors:
        print(f"[警告] 找不到节点: {int8_name}")
        continue
    
    inp_tensor = tensors[0]
    out_tensor = gs.Variable(name=target_name, dtype=np.float32)
    
    # 插入 Cast 节点
    cast_node = gs.Node(op="Cast", inputs=[inp_tensor], outputs=[out_tensor], attrs={"to": 1})
    graph.nodes.append(cast_node)
    graph.outputs.append(out_tensor)
    count += 1
    print(f"[成功] 节点 {int8_name} -> FP32 -> {target_name}")

if count == 0:
    print("失败：未找到任何目标节点。")
    exit(1)

# 4. 清理图
print("正在清理模型...")
try:
    graph.cleanup().toposort()
except Exception as e:
    print(f"[警告] 清理过程跳过 (非致命): {e}")

# 5. 【关键修改】导出并强制降级 IR 版本
print("正在执行版本降级 (IR 12 -> 10)...")
model_proto = gs.export_onnx(graph)
model_proto.ir_version = 10  # 强制设置为您环境支持的最大版本

onnx.save(model_proto, output_model)
print(f"\n✅ 兼容版模型已保存: {output_model}")