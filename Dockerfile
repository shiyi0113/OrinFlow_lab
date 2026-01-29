FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

LABEL maintainer="chenrong <shiyileshengxia@gmail.com>"
LABEL description="Environment for AI Edge Deployment (PC)"

# 替换源为国内源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip install --upgrade pip
# 环境变量
ENV PIP_CONSTRAINT=""
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib/:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib/"

# 安装基础系统工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    rm -rf /var/lib/apt/lists/*

# 默认启动命令
CMD ["/bin/bash"]