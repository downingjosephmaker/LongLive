#!/bin/bash
# LongLive Docker 部署脚本
# 所有文件都在项目目录中，通过 WSL2 挂载路径访问
# 使用方法: ./shell.sh [all|build|up|down|logs|restart]

echo -e "\033[32;40m========== LongLive Docker 部署 ==========\033[0m"

# 配置
PROJECT_DIR="/mnt/d/MyWork/Project/My/LongLive"
NETWORK_NAME="longlive-network"

# 确保项目中的必要目录存在
mkdir -p "$PROJECT_DIR"/{wan_models,inference_prompts,output}

# 进入项目目录
cd "$PROJECT_DIR"

case $1 in
"down")
    echo -e "\033[31m🛑 停止服务...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml down
    ;;

"up")
    echo -e "\033[32m🚀 启动服务...\033[0m"
    docker network create "$NETWORK_NAME" 2>/dev/null || true
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml up -d
    echo -e "\033[32m✓ 服务已启动\033[0m"
    docker ps --filter "name=longlive"
    ;;

"build")
    echo -e "\033[33m🔨 构建镜像...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml build --no-cache
    echo -e "\033[32m✓ 镜像构建完成\033[0m"
    ;;

"rebuild")
    echo -e "\033[33m🔨 重新构建镜像（使用缓存）...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml build
    echo -e "\033[32m✓ 镜像构建完成\033[0m"
    ;;

"restart")
    echo -e "\033[33m🔄 重启服务...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml restart
    echo -e "\033[32m✓ 服务已重启\033[0m"
    docker ps --filter "name=longlive"
    ;;

"quick")
    echo -e "\033[33m⚡ 快速更新（复制yml + 重启容器，不重新构建镜像）...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml down
    echo -e "\033[32m🚀 启动服务...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml up -d
    echo -e "\033[32m✓ 快速更新完成！\033[0m"
    docker ps --filter "name=longlive"
    ;;

"logs")
    echo -e "\033[36m📋 查看日志 (Ctrl+C 退出)...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml logs -f
    ;;

"all")
    echo -e "\033[32m🚀 完整部署流程...\033[0m"

    echo -e "\033[33m🌐 检查网络...\033[0m"
    docker network create "$NETWORK_NAME" 2>/dev/null || true

    echo -e "\033[31m🛑 停止旧容器...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml down

    echo -e "\033[33m🔨 构建镜像...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml build --no-cache

    echo -e "\033[32m🚀 启动服务...\033[0m"
    docker-compose -f /home/docker_yml/longlive/docker-compose.yml up -d

    echo -e "\033[33m🧹 清理缓存...\033[0m"
    docker system prune -f

    echo -e "\033[32m✓ 部署完成！\033[0m"
    docker ps --filter "name=longlive"
    echo ""
    echo -e "\033[36m📂 挂载目录:\033[0m"
    echo "   wan_models           → $PROJECT_DIR/wan_models"
    echo "   inference_prompts→ $PROJECT_DIR/inference_prompts"
    echo "   output           → $PROJECT_DIR/output"
    echo ""
    echo -e "\033[36m📋 查看日志: ./shell.sh logs\033[0m"
    ;;

*)
    echo -e "\033[31m========== 使用方法 ==========\033[0m"
    echo "  ./shell.sh all      - 完整部署（停止+构建+启动）"
    echo "  ./shell.sh build    - 构建镜像（无缓存）"
    echo "  ./shell.sh rebuild  - 构建镜像（使用缓存）"
    echo "  ./shell.sh up       - 启动服务"
    echo "  ./shell.sh down     - 停止服务"
    echo "  ./shell.sh restart  - 重启服务"
    echo "  ./shell.sh quick    - 快速更新（只更新yml+重启，不构建镜像）"
    echo "  ./shell.sh logs     - 查看日志"
    echo -e "\033[31m================================\033[0m"
    exit 1
    ;;
esac
