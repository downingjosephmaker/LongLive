FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 使用清华源加速 apt
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# 系统依赖：构建工具 + Pillow/opencv 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 步骤1：安装 PyTorch 2.9.1+cu128 及全部 CUDA 依赖 (支持 RTX 5090 Blackwell)
# 从本地 wheel 文件安装（已预下载，避免网络问题）
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    /tmp/wheels/torch-2.9.1+cu128-*.whl \
    /tmp/wheels/torchvision-0.24.1+cu128-*.whl \
    /tmp/wheels/torchaudio-2.9.1+cu128-*.whl \
    /tmp/wheels/nvidia_cublas_cu12-*.whl \
    /tmp/wheels/nvidia_cuda_cupti_cu12-*.whl \
    /tmp/wheels/nvidia_cuda_nvrtc_cu12-*.whl \
    /tmp/wheels/nvidia_cuda_runtime_cu12-*.whl \
    /tmp/wheels/nvidia_cudnn_cu12-*.whl \
    /tmp/wheels/nvidia_cufft_cu12-*.whl \
    /tmp/wheels/nvidia_cufile_cu12-*.whl \
    /tmp/wheels/nvidia_curand_cu12-*.whl \
    /tmp/wheels/nvidia_cusolver_cu12-*.whl \
    /tmp/wheels/nvidia_cusparse_cu12-*.whl \
    /tmp/wheels/nvidia_cusparselt_cu12-*.whl \
    /tmp/wheels/nvidia_nccl_cu12-*.whl \
    /tmp/wheels/nvidia_nvjitlink_cu12-*.whl \
    /tmp/wheels/nvidia_nvshmem_cu12-*.whl \
    /tmp/wheels/nvidia_nvtx_cu12-*.whl \
    /tmp/wheels/triton-*.whl \
    /tmp/wheels/torchao-*.whl \
    /tmp/wheels/filelock-*.whl \
    /tmp/wheels/fsspec-*.whl \
    /tmp/wheels/jinja2-*.whl \
    /tmp/wheels/markupsafe-*.whl \
    /tmp/wheels/mpmath-*.whl \
    /tmp/wheels/networkx-*.whl \
    /tmp/wheels/setuptools-*.whl \
    /tmp/wheels/sympy-*.whl \
    /tmp/wheels/typing_extensions-*.whl && \
    rm -rf /tmp/wheels

# 步骤2：安装 requirements.txt 中的依赖（使用清华PyPI源）
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt || \
    pip install --no-cache-dir --timeout=600 --retries=5 \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt --ignore-installed torch torchvision torchaudio

# 步骤3：安装 Web 服务依赖
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    uvicorn fastapi python-multipart

# 注意：flash-attn 在容器中编译容易失败，如需使用请预先编译或使用预编译 wheel
# PyTorch 2.6.0 的原生 SDPA 在 RTX 5090 上性能已经很好

# 项目代码
COPY . .

# Runtime directories
RUN mkdir -p /app/wan_models /inference_prompts /output /tmp/longlive_uploads

# API port
EXPOSE 19521

# Default: start API server
ENTRYPOINT ["python", "api_server.py"]
