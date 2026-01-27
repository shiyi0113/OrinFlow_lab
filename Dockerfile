FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /workspace

LABEL maintainer="shiyi <shiyileshengxia@gmail.com>"
LABEL description="Environment for AI Edge Deployment (PC)"

# 替换源为国内源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip install --upgrade pip

ENV PIP_CONSTRAINT=""
# 安装基础系统工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    curl &&\
    rm -rf /var/lib/apt/lists/*
# uv
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

#---- 导出环境 ----
RUN uv venv /opt/envs/venv_export
RUN uv pip install --python /opt/envs/venv_export --no-cache-dir\
    ultralytics \
    pandas \
    onnx==0.15.0\
    onnxslim\
    onnxrumtime-gpu==1.17.3

#---- 量化环境 ----
RUN uv venv --system-site-packages /opt/envs/venv_quant
RUN uv pip install --python /opt/envs/venv_quant --no-cache-dir\
    "onnx<2" \
    "nvidia-modelopt[all]" \
    onnxmltools==1.13.0 \
    --extra-index-url https://pypi.nvidia.com
RUN uv pip install --python /opt/envs/venv_quant --no-cache-dir \
    onnxruntime-gpu==1.19.2 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu"
# 快捷方式
RUN echo "alias use_export='source /opt/envs/venv_export/bin/activate'" >> ~/.bashrc && \
    echo "alias use_quant='source /opt/envs/venv_quant/bin/activate'" >> ~/.bashrc
# 默认启动命令
CMD ["/bin/bash"]