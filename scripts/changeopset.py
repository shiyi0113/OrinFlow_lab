import onnx

# 读取量化后的模型
model_path = "models/onnx/optimized/yolo26x_INT8.onnx"
model = onnx.load(model_path)

# 强行修改 Opset 版本号
current_version = model.opset_import[0].version
print(f"当前 Opset 版本: {current_version}")

if current_version > 19:
    print("检测到版本过高，正在降级到 Opset 19...")
    model.opset_import[0].version = 19
    # 重新保存
    onnx.save(model, model_path)
    print("降级完成。")