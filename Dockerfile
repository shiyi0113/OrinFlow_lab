# 1. 指定基础镜像
FROM nvcr.io/nvidia/tensorrt:24.08-py3

# 2. 设置环境变量
ENV PIP_CONSTRAINT=""
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu"

# 3. 设置工作目录
WORKDIR /workspace/OrinFlow_Lab

# 4. 安装系统依赖
# 使用 --no-install-recommends 减小镜像体积，安装完后清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 5. 安装 Python 库
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -U "nvidia-modelopt[all]" && \
    pip install --no-cache-dir ultralytics onnxruntime-gpu onnxslim

# 默认启动命令
CMD ["/bin/bash"]