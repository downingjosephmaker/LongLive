FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 使用清华源加速 apt
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制所有预下载的 wheel（纯离线安装，不需要 requirements.txt）
COPY wheels/ /tmp/wheels/

# ============================================================
# 全部离线安装，--no-deps 禁止 pip 联网
# ============================================================

# 步骤1：PyTorch 2.9.1+cu128（--no-deps 跳过依赖检查）
RUN pip install --no-cache-dir --no-deps \
    /tmp/wheels/torch-2.9.1+cu128-*.whl \
    /tmp/wheels/torchvision-0.24.1+cu128-*.whl \
    /tmp/wheels/torchaudio-2.9.1+cu128-*.whl

# 步骤1.1：NVIDIA CUDA 依赖 + PyTorch 基础库
RUN pip install --no-cache-dir --no-deps \
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
    /tmp/wheels/setuptools-*.whl \
    /tmp/wheels/filelock-*.whl \
    /tmp/wheels/fsspec-*.whl \
    /tmp/wheels/jinja2-*.whl \
    /tmp/wheels/markupsafe-*.whl \
    /tmp/wheels/mpmath-*.whl \
    /tmp/wheels/networkx-*.whl \
    /tmp/wheels/sympy-*.whl \
    /tmp/wheels/typing_extensions-*.whl

# 步骤1.5：flash-attn（RTX 5090 Blackwell sm_120）
RUN pip install --no-cache-dir --no-deps \
    /tmp/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 步骤1.7：CLIP（本地源码，--no-build-isolation 使用已有 setuptools）
RUN pip install --no-cache-dir --no-deps --no-build-isolation /tmp/wheels/CLIP/ \
    && rm -rf /tmp/wheels/CLIP

# 步骤2：纯离线批量安装所有 .whl（跳过 tar.gz 避免 antlr4 构建失败中断安装）
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl

# 步骤2.1：安装 antlr4（omegaconf 的依赖）
RUN pip install --no-cache-dir antlr4-python3-runtime==4.9.3

# 清理
RUN rm -rf /tmp/wheels

# 步骤3：安装 Web 服务依赖（这3个小包很快）
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    uvicorn fastapi python-multipart

# 项目代码
COPY . .

# Runtime directories
RUN mkdir -p /app/wan_models /inference_prompts /output /tmp/longlive_uploads

# API port
EXPOSE 19521

# Default: start API server
ENTRYPOINT ["python", "api_server.py"]
