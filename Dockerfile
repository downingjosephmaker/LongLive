FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 使用清华源加速 apt
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# 最小系统依赖（不需要 CUDA toolkit / nvcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖 - 使用清华源
COPY requirements.txt .
# pip 配置：超时 300s，重试 5 次
RUN pip config set global.timeout 300 && \
    pip config set global.retries 5

# Python 依赖 - 清华源为主，pytorch 源为辅
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uvicorn fastapi python-multipart

# flash-attn: 尝试编译，失败则跳过（代码有 fallback 用 torch SDPA）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation || \
    echo "[WARN] flash-attn compilation failed, will use torch SDPA fallback"

# 项目代码
COPY . .

# Runtime directories
RUN mkdir -p /models /prompts /output /tmp/longlive_uploads

# API port
EXPOSE 19521

# Default: start API server
ENTRYPOINT ["python", "api_server.py"]
