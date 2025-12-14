#!/usr/bin/env python3
"""
VoxCPM LoRA TTS API 启动脚本

提供灵活的配置选项来启动 API 服务
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="启动 VoxCPM LoRA TTS API 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python start_api.py                          # 默认配置 (0.0.0.0:8000)
  python start_api.py --port 8080              # 指定端口
  python start_api.py --host 127.0.0.1         # 仅本地访问
  python start_api.py --workers 4              # 多进程模式
  python start_api.py --dev                    # 开发模式 (启用热重载)
        """
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务器地址 (默认: 0.0.0.0，所有网络接口)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (默认: 1)"
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="开发模式，启用自动重载"
    )

    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="日志级别 (默认: info)"
    )

    parser.add_argument(
        "--ssl-keyfile",
        help="SSL 私钥文件路径 (启用 HTTPS)"
    )

    parser.add_argument(
        "--ssl-certfile",
        help="SSL 证书文件路径 (启用 HTTPS)"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs("api_outputs", exist_ok=True)

    # 配置参数
    config = {
        "app": "api_server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "reload": args.dev,
    }

    # 多进程模式
    if args.workers > 1:
        config["workers"] = args.workers
        # 多进程模式下禁用自动重载
        config["reload"] = False

    # SSL 配置
    if args.ssl_keyfile and args.ssl_certfile:
        config["ssl_keyfile"] = args.ssl_keyfile
        config["ssl_certfile"] = args.ssl_certfile
        protocol = "https"
    else:
        protocol = "http"

    # 显示启动信息
    print("=" * 60)
    print("🚀 启动 VoxCPM LoRA TTS API 服务")
    print("=" * 60)
    print(f"📍 服务地址: {protocol}://{args.host}:{args.port}")
    print(f"📝 API 文档: {protocol}://{args.host}:{args.port}/docs")
    print(f"📊 OpenAPI: {protocol}://{args.host}:{args.port}/openapi.json")
    print(f"🔧 工作进程: {args.workers}")
    print(f"📋 日志级别: {args.log_level}")
    if args.dev:
        print("🛠️  开发模式: 已启用")
    if protocol == "https":
        print("🔒 HTTPS: 已启用")
    print("=" * 60)

    # 启动服务
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n✋ 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()