#!/usr/bin/env python3
"""
网络配置测试脚本

帮助用户检查网络配置和连接状态
"""

import socket
import requests
import subprocess
import platform

def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        # 创建一个连接到外网的socket来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def get_network_interfaces():
    """获取网络接口信息"""
    try:
        system = platform.system().lower()
        if system == "windows":
            result = subprocess.run(["ipconfig"], capture_output=True, text=True)
        else:
            result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
        return result.stdout
    except Exception:
        return "无法获取网络接口信息"

def test_port(host, port):
    """测试端口是否可访问"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def test_api_endpoint(url):
    """测试API端点是否可访问"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("🌐 VoxCPM API 网络配置测试")
    print("=" * 60)

    # 获取本机IP
    local_ip = get_local_ip()
    print(f"🖥️  本机局域网IP: {local_ip or '无法获取'}")

    # 获取所有网络接口
    print("\n📡 网络接口信息:")
    print("-" * 40)
    interfaces = get_network_interfaces()
    # 只显示前几行
    for line in interfaces.split('\n')[:20]:
        print(line)
    if len(interfaces.split('\n')) > 20:
        print("... (更多信息)")

    # 常见的端口测试
    port = 8000
    hosts_to_test = [
        ("localhost", "本地回环"),
        ("127.0.0.1", "本地IP"),
    ]

    if local_ip:
        hosts_to_test.append((local_ip, "局域网IP"))

    print(f"\n🔌 端口 {port} 连接测试:")
    print("-" * 40)
    for host, desc in hosts_to_test:
        status = "✅ 可连接" if test_port(host, port) else "❌ 无法连接"
        print(f"  {host:15} ({desc}): {status}")

    # API端点测试
    print(f"\n🚀 API端点测试:")
    print("-" * 40)

    base_urls = [
        ("http://localhost:8000", "本地访问"),
    ]

    if local_ip:
        base_urls.append((f"http://{local_ip}:8000", "局域网访问"))

    for url, desc in base_urls:
        is_available, result = test_api_endpoint(url)
        if is_available:
            print(f"  {desc:10} ({url}): ✅ 正常")
            print(f"    - 状态: {result.get('status', 'unknown')}")
            print(f"    - 模型: {'已加载' if result.get('model_loaded') else '未加载'}")
        else:
            print(f"  {desc:10} ({url}): ❌ 无法访问")
            print(f"    - 错误: {result}")

    print(f"\n📋 访问建议:")
    print("-" * 40)
    print("1. 本地开发: 使用 http://localhost:8000")
    if local_ip:
        print(f"2. 局域网访问: 使用 http://{local_ip}:8000")
    print("3. API文档: http://[您的IP]:8000/docs")
    print("\n🔧 如果无法连接，请检查:")
    print("   - 防火墙设置")
    print("   - 是否已启动API服务")
    print("   - 端口是否被占用")
    print("   - 网络连接状态")

if __name__ == "__main__":
    main()