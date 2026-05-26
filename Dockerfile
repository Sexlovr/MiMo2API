# 使用 Python 3.10 轻量版作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建 /data 目录并设置权限（用于 Hugging Face 存储挂载）
RUN mkdir -p /data && chmod 777 /data

# 设置环境变量
ENV DATA_DIR=/data
ENV PORT=7860

# 暴露端口（Railway 会自动映射，Hugging Face 需要 7860）
EXPOSE 7860

# 启动命令
CMD ["python", "main.py"]
