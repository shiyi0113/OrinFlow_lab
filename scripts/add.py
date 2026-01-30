import onnx_graphsurgeon as gs
import onnx
import numpy as np

# 1. 设置路径
input_model = "models/onnx/optimized/yolo26x_INT8.onnx"
output_model = "yolo26x_INT8_READY_TO_RUN_v4.onnx"

# 2. 定义映射
node_map = {
    "/model.16/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.16/cv2/act/Mul_output_0",
    "/model.19/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.19/cv2/act/Mul_output_0",
    "/model.22/cv2/act/Mul_output_0_DequantizeLinear_Output": "/model.22/cv2/act/Mul_output_0"
}

print(f"加载模型: {input_model}...")
graph = gs.import_onnx(onnx.load(input_model))

# 3. 【核心修正】先清除所有旧输出！
# 这样 ONNX Runtime 就不会去跑原来那些会报错的 FP16 输出了
print(f"清理前输出数量: {len(graph.outputs)}")
graph.outputs.clear()
print("已清除旧输出，准备添加调试输出...")

# 4. 防止名字冲突
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
    
    # 添加为新输出
    graph.outputs.append(out_tensor)
    count += 1
    print(f"[成功] 节点 {int8_name} -> FP32 -> {target_name}")

if count == 0:
    print("失败：未找到任何目标节点。")
    exit(1)

# 5. 清理图 (这一步会自动把不连接到新输出的后续节点全部剪枝删掉)
print("正在清理并截断模型...")
try:
    graph.cleanup().toposort()
except Exception as e:
    print(f"[警告] {e}")

# 6. 强制降级 IR 版本
print("正在执行版本降级 (IR 12 -> 10)...")
model_proto = gs.export_onnx(graph)
model_proto.ir_version = 10 

onnx.save(model_proto, output_model)
print(f"\n✅ 纯净版调试模型已保存: {output_model}")