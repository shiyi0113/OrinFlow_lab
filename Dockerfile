FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

LABEL maintainer="chenrong <shiyileshengxia@gmail.com>"
LABEL description="Environment for AI Edge Deployment (PC)"

# 替换源为国内源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装 Python 和系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# pip 配置国内源
RUN python -m pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ENV PIP_CONSTRAINT=""
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu"

# 安装 PyTorch GPU (Nightly版本支持RTX5060显卡)
RUN pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# 安装 NVIDIA ModelOpt + ONNX Runtime GPU
RUN pip install nvidia-modelopt[all] onnxruntime-gpu onnx \
    --extra-index-url https://pypi.nvidia.com

# 默认启动命令
CMD ["/bin/bash"]
