#!/usr/bin/env python3
"""
异步API测试脚本

演示完整的异步语音合成工作流程：
1. 提交异步任务
2. 定期查询任务状态
3. 下载生成的音频文件
"""

import requests
import time
import json
import os
from datetime import datetime

# API配置
BASE_URL = "http://localhost:8000"  # 本地访问
# BASE_URL = "http://192.168.1.100:8000"  # 局域网访问 (请替换为实际IP)

def print_separator(title):
    """打印分隔线"""
    print("=" * 60)
    print(f"🔹 {title}")
    print("=" * 60)

def test_api_health():
    """测试API健康状态"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API状态: {data['status']}")
            print(f"   模型已加载: {'是' if data['model_loaded'] else '否'}")
            return True
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API: {e}")
        return False

def list_available_loras():
    """列出可用的LoRA模型"""
    try:
        response = requests.get(f"{BASE_URL}/loras", timeout=60)
        if response.status_code == 200:
            data = response.json()
            loras = data.get('loras', [])
            print(f"📋 可用LoRA模型 ({len(loras)}个):")
            for i, lora in enumerate(loras, 1):
                print(f"   {i}. {lora}")
            return loras
        else:
            print(f"❌ 获取LoRA列表失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 获取LoRA列表出错: {e}")
        return []

def submit_async_task(text, lora_name=None, steps=15, cfg_scale=2.0):
    """提交异步语音合成任务"""
    print_separator("提交异步任务")

    request_data = {
        "text": text,
        "lora_name": lora_name,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "seed": int(time.time()),  # 使用当前时间作为随机种子
        "async_mode": True  # 异步模式
    }

    print(f"📝 合成文本: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"🎭 LoRA模型: {lora_name or '无'}")
    print(f"⚙️  参数: CFG={cfg_scale}, Steps={steps}")
    print(f"🔄 模式: 异步")

    try:
        response = requests.post(f"{BASE_URL}/synthesize", json=request_data, timeout=120)

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "submitted":
                task_id = result["task_id"]
                estimated_time = result.get("estimated_time", 0)
                print(f"✅ 任务提交成功!")
                print(f"   任务ID: {task_id}")
                print(f"   预计时间: {estimated_time}秒")
                print(f"   初始进度: {result.get('progress', 0)*100:.1f}%")
                return task_id
            else:
                print(f"❌ 任务提交失败: {result.get('message', '未知错误')}")
                return None
        else:
            error_detail = response.json().get("detail", "未知错误")
            print(f"❌ 请求失败 ({response.status_code}): {error_detail}")
            return None

    except Exception as e:
        print(f"❌ 提交任务时出错: {e}")
        return None

def poll_task_status(task_id, interval=3, timeout=600):
    """轮询任务状态"""
    print_separator(f"监控任务状态 (ID: {task_id})")

    start_time = time.time()
    last_progress = -1

    print("⏱️  开始监控任务进度...")
    print(f"   查询间隔: {interval}秒")
    print(f"   超时时间: {timeout}秒")
    print()

    while True:
        try:
            response = requests.get(f"{BASE_URL}/task/{task_id}", timeout=10)

            if response.status_code == 200:
                task_status = response.json()
                status = task_status["status"]
                progress = task_status.get("progress", 0) * 100
                message = task_status.get("message", "")
                created_at = task_status.get("created_at", "")
                updated_at = task_status.get("updated_at", "")

                # 只在进度更新时显示
                if progress != last_progress or status in ["completed", "failed"]:
                    elapsed = int(time.time() - start_time)
                    print(f"⏰ {elapsed:3d}s | 状态: {status:10s} | 进度: {progress:5.1f}% | {message}")

                # 检查任务是否完成
                if status == "completed":
                    print(f"\n🎉 任务完成!")
                    print(f"   处理时间: {int(time.time() - start_time)}秒")
                    print(f"   音频文件: {task_status.get('audio_path', '未知')}")
                    return task_status

                elif status == "failed":
                    error_msg = task_status.get("error", "未知错误")
                    print(f"\n❌ 任务失败: {error_msg}")
                    return None

                last_progress = progress

            else:
                print(f"❌ 查询任务状态失败: {response.status_code}")

            # 检查超时
            if time.time() - start_time > timeout:
                print(f"\n⏰ 任务监控超时 ({timeout}秒)")
                return None

            time.sleep(interval)

        except Exception as e:
            print(f"❌ 查询状态时出错: {e}")
            time.sleep(interval)

def download_audio_file(audio_path, save_path=None):
    """下载音频文件"""
    print_separator("下载音频文件")

    if not audio_path:
        print("❌ 没有音频文件可下载")
        return False

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"tts_output_{timestamp}.mp3"

    filename = os.path.basename(audio_path)
    download_url = f"{BASE_URL}/download/{filename}"

    print(f"📥 下载文件: {filename}")
    print(f"   保存路径: {save_path}")
    print(f"   下载地址: {download_url}")

    try:
        response = requests.get(download_url, timeout=30, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   下载进度: {progress:.1f}%", end='', flush=True)

            print()  # 换行
            file_size = os.path.getsize(save_path)
            print(f"✅ 下载完成!")
            print(f"   文件大小: {file_size:,} 字节 ({file_size/1024/1024:.1f} MB)")
            return True

        else:
            print(f"❌ 下载失败: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 下载时出错: {e}")
        return False

def test_task_management():
    """测试任务管理功能"""
    print_separator("任务管理测试")

    try:
        # 获取所有任务
        response = requests.get(f"{BASE_URL}/tasks", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"📊 任务统计:")
            print(f"   总任务数: {data['total']}")
            print(f"   正在处理: {data['processing']}/{data['max_concurrent']}")

            # 显示最近的任务
            if data['tasks']:
                print(f"\n📋 最近任务 (最多5个):")
                for task in data['tasks'][:5]:
                    task_id = task['task_id']
                    status = task['status']
                    progress = task.get('progress', 0) * 100
                    created = task.get('created_at', '')[:19]  # 去掉毫秒
                    print(f"   {task_id} | {status:10s} | {progress:5.1f}% | {created}")

    except Exception as e:
        print(f"❌ 任务管理测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 VoxCPM 异步API测试脚本")
    print("🎯 目标: 测试完整的异步语音合成工作流程\n")

    # 1. 测试API连接
    if not test_api_health():
        print("❌ API服务不可用，请确保服务已启动")
        return

    # 2. 查看可用LoRA模型
    loras = list_available_loras()

    # 3. 测试任务管理
    test_task_management()

    # 4. 提交异步任务
    # 使用较长的文本来测试异步处理
    test_texts = [
        {
            "text": "欢迎使用VoxCPM语音合成系统。这是一个测试文本，用于演示异步语音合成功能。" * 3,
            "lora": "lora1" if "lora1" in loras else None,
            "steps": 12
        },
        {
            "text": "Hello, this is a test of the VoxCPM async TTS system. " * 5,
            "lora": None,
            "steps": 10
        }
    ]

    # 选择第一个测试用例
    test_case = test_texts[0]

    task_id = submit_async_task(
        text=test_case["text"],
        lora_name=test_case["lora"],
        steps=test_case["steps"]
    )

    if not task_id:
        print("❌ 无法提交任务，测试终止")
        return

    # 5. 监控任务状态
    task_result = poll_task_status(task_id, interval=3, timeout=300)

    if not task_result:
        print("❌ 任务未成功完成")
        return

    # 6. 下载音频文件
    audio_path = task_result.get("audio_path")
    success = download_audio_file(audio_path)

    if success:
        print("\n" + "=" * 60)
        print("🎉 异步API测试完成!")
        print("✅ 所有步骤都成功执行")
        print("✅ 音频文件已下载")
        print("=" * 60)
    else:
        print("\n❌ 音频下载失败")

    # 7. 最终任务状态检查
    print_separator("最终状态检查")
    test_task_management()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现未预期的错误: {e}")
        import traceback
        traceback.print_exc()