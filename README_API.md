# VoxCPM LoRA TTS RESTful API

这是一个基于 FastAPI 构建的 VoxCPM LoRA 文本转语音 RESTful API 服务，支持通过 HTTP 请求进行语音合成，包括 LoRA 模型加载和声音克隆功能。

## 功能特性

- 🎵 **文本转语音**: 支持高质量的文字转语音合成
- 🎭 **LoRA 模型支持**: 动态加载和切换不同的 LoRA 微调模型
- 🎤 **声音克隆**: 支持使用参考音频进行声音克隆
- ⚡ **异步处理**: 支持长文本异步处理，避免HTTP连接超时
- 📊 **任务管理**: 实时任务状态跟踪和进度查询
- 🎛️ **参数控制**: 可调节 CFG Scale、推理步数、随机种子等参数
- 📁 **文件管理**: 自动生成和管理音频文件，支持 MP3 格式输出
- 🔄 **热加载**: 智能模型缓存和热切换，提高响应速度
- 🌐 **网络接口**: 支持所有网络接口访问，便于局域网使用

## 安装依赖

```bash
pip install -r requirements_api.txt
```

## 启动服务

### 方式一：使用启动脚本（推荐）

```bash
# 默认配置 (所有网络接口:8000)
python start_api.py

# 指定端口
python start_api.py --port 8080

# 仅本地访问
python start_api.py --host 127.0.0.1

# 开发模式 (启用热重载)
python start_api.py --dev

# 多进程模式
python start_api.py --workers 4

# HTTPS 模式
python start_api.py --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### 方式二：直接运行

```bash
python api_server.py
```

**默认配置下，服务将在所有网络接口上监听端口 8000：**
- 本地访问: `http://localhost:8000` 或 `http://127.0.0.1:8000`
- 局域网访问: `http://[您的IP地址]:8000` (例如: `http://192.168.1.100:8000`)
- API 文档: `http://[您的IP地址]:8000/docs`

### 网络配置测试

运行测试脚本来检查网络配置：

```bash
python test_network.py
```

这个脚本会：
- 显示您的局域网IP地址
- 测试端口连接状态
- 验证API端点可访问性
- 提供访问建议

## 测试API

### 完整异步测试

```bash
# 运行完整测试脚本，演示异步工作流程
python test_async_api.py
```

该脚本会：
- ✅ 测试API连接和健康状态
- 📋 列出可用的LoRA模型
- 📊 显示任务管理功能
- 🔄 提交异步任务并监控进度
- 📥 下载生成的音频文件

## API 端点

### 1. 健康检查
```
GET /health
```

检查 API 服务状态。

**响应示例:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 列出可用 LoRA 模型
```
GET /loras
```

获取所有可用的 LoRA 模型列表。

**响应示例:**
```json
{
  "loras": ["lora_model_1", "lora_model_2", "lora_model_3"],
  "count": 3
}
```

### 3. 语音合成
```
POST /synthesize
```

核心接口，用于生成语音。支持同步和异步两种模式。

**请求体:**
```json
{
  "text": "要合成的文本内容",
  "lora_name": "lora_model_name",  // 可选，不使用 LoRA 则为 null 或 "None"
  "cfg_scale": 2.0,               // 可选，默认 2.0
  "steps": 10,                    // 可选，默认 10
  "seed": -1,                     // 可选，-1 为随机
  "ref_audio_path": "path/to/ref.wav",  // 可选，声音克隆
  "ref_text": "参考音频的文本内容",       // 可选，声音克隆
  "async_mode": true              // 可选，默认 true，是否使用异步模式
}
```

**异步模式响应 (默认):**
```json
{
  "task_id": "abc12345",
  "status": "submitted",
  "message": "任务已提交，请使用task_id查询处理状态",
  "estimated_time": 45,
  "progress": 0.0
}
```

**同步模式响应:**
```json
{
  "task_id": "abc12345",
  "status": "success",
  "message": "Speech synthesized successfully",
  "audio_path": "api_outputs/tts_abc12345_1640995200.mp3",
  "sample_rate": 44100
}
```

### 3.1 任务状态查询
```
GET /task/{task_id}
```

查询特定任务的状态和进度。

**响应示例:**
```json
{
  "task_id": "abc12345",
  "status": "processing",
  "message": "生成音频中...",
  "progress": 0.6,
  "created_at": "2023-12-14T12:00:00",
  "updated_at": "2023-12-14T12:01:30",
  "estimated_time": 45,
  "audio_path": null
}
```

**任务状态说明:**
- `pending`: 等待处理
- `processing`: 正在处理
- `completed`: 处理完成
- `failed`: 处理失败

### 3.2 任务列表
```
GET /tasks?status=processing&limit=10
```

获取任务列表，支持状态过滤。

**查询参数:**
- `status`: 可选，过滤任务状态 (pending/processing/completed/failed)
- `limit`: 可选，限制返回数量，默认50

**响应示例:**
```json
{
  "tasks": [...],
  "total": 25,
  "processing": 2,
  "max_concurrent": 2
}
```

### 4. 下载音频文件
```
GET /download/{filename}
```

下载生成的音频文件。

### 5. 清理旧文件
```
DELETE /cleanup
```

清理超过 1 小时的旧音频文件。

**响应示例:**
```json
{
  "message": "Cleanup completed",
  "deleted_count": 5
}
```

## 使用示例

### Python 客户端

#### 异步工作流程（推荐用于长文本）

```python
import requests
import time
import json

# 1. 提交异步任务
response = requests.post("http://localhost:8000/synthesize", json={
    "text": "这是一个很长的文本内容，需要进行异步处理..." * 10,
    "lora_name": "lora1",
    "cfg_scale": 2.0,
    "steps": 15,
    "seed": 42,
    "async_mode": True  # 异步模式（默认）
})

result = response.json()
if result["status"] == "submitted":
    task_id = result["task_id"]
    print(f"任务已提交: {task_id}")
    print(f"预计处理时间: {result['estimated_time']}秒")

    # 2. 轮询任务状态
    while True:
        status_response = requests.get(f"http://localhost:8000/task/{task_id}")
        task_status = status_response.json()

        print(f"任务状态: {task_status['status']}")
        print(f"进度: {task_status.get('progress', 0)*100:.1f}%")
        print(f"消息: {task_status.get('message', '')}")

        if task_status["status"] == "completed":
            print(f"任务完成! 音频文件: {task_status['audio_path']}")
            # 下载音频文件
            filename = task_status['audio_path'].split('/')[-1]
            audio_response = requests.get(f"http://localhost:8000/download/{filename}")
            with open("async_output.mp3", "wb") as f:
                f.write(audio_response.content)
            break
        elif task_status["status"] == "failed":
            print(f"任务失败: {task_status.get('error', '未知错误')}")
            break

        time.sleep(2)  # 每2秒查询一次
```

#### 同步工作流程（适用于短文本）

```python
import requests

# 同步模式 - 直接等待结果
response = requests.post("http://localhost:8000/synthesize", json={
    "text": "Hello, this is a test of VoxCPM TTS API.",
    "lora_name": "lora1",
    "cfg_scale": 2.0,
    "steps": 10,
    "async_mode": False  # 同步模式
})

result = response.json()
if result["status"] == "success":
    print(f"Audio generated: {result['audio_path']}")
    # 下载音频文件
    audio_response = requests.get(f"http://localhost:8000/download/{result['audio_path'].split('/')[-1]}")
    with open("sync_output.mp3", "wb") as f:
        f.write(audio_response.content)
```

#### 任务管理示例

```python
# 查看所有任务
response = requests.get("http://localhost:8000/tasks")
all_tasks = response.json()
print(f"总任务数: {all_tasks['total']}")
print(f"正在处理: {all_tasks['processing']}/{all_tasks['max_concurrent']}")

# 查看正在处理的任务
response = requests.get("http://localhost:8000/tasks?status=processing")
processing_tasks = response.json()
for task in processing_tasks['tasks']:
    print(f"任务ID: {task['task_id']}, 进度: {task['progress']*100:.1f}%")

# 查看可用 LoRA 模型
response = requests.get("http://localhost:8000/loras")
loras = response.json()
print(f"Available LoRAs: {loras['loras']}")

# 声音克隆示例
response = requests.post("http://localhost:8000/synthesize", json={
    "text": "这是使用声音克隆技术合成的语音。",
    "ref_audio_path": "reference_audio.wav",
    "ref_text": "这是参考音频的内容",
    "cfg_scale": 2.0,
    "steps": 20
})
```

### curl 示例

#### 异步工作流程

```bash
# 1. 提交异步任务
TASK_ID=$(curl -s -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一个很长的文本内容，需要进行异步处理..." * 5,
    "lora_name": "lora1",
    "async_mode": true
  }' | jq -r '.task_id')

echo "任务已提交: $TASK_ID"

# 2. 轮询任务状态
while true; do
  STATUS=$(curl -s "http://localhost:8000/task/$TASK_ID" | jq -r '.status')
  PROGRESS=$(curl -s "http://localhost:8000/task/$TASK_ID" | jq -r '.progress')
  MESSAGE=$(curl -s "http://localhost:8000/task/$TASK_ID" | jq -r '.message')

  echo "状态: $STATUS, 进度: $(echo "$PROGRESS * 100" | bc)%, 消息: $MESSAGE"

  if [ "$STATUS" = "completed" ]; then
    AUDIO_PATH=$(curl -s "http://localhost:8000/task/$TASK_ID" | jq -r '.audio_path')
    echo "任务完成! 音频文件: $AUDIO_PATH"
    # 下载音频文件
    curl -X GET "http://localhost:8000/download/$(basename $AUDIO_PATH)" -o async_output.mp3
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "任务失败"
    break
  fi

  sleep 3
done
```

#### 同步工作流程

```bash
# 同步模式 - 直接等待结果
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of VoxCPM TTS API.",
    "lora_name": "lora1",
    "cfg_scale": 2.0,
    "steps": 10,
    "async_mode": false
  }'

# 列出 LoRA 模型
curl -X GET "http://localhost:8000/loras"
# 响应: {"loras":["lora1","20251214_173819/checkpoints/step_0000200",...], "count":6}

# 查看任务状态
curl -X GET "http://localhost:8000/task/abc12345"

# 查看所有任务
curl -X GET "http://localhost:8000/tasks"

# 查看正在处理的任务
curl -X GET "http://localhost:8000/tasks?status=processing"

# 下载音频文件 (替换为实际文件名)
curl -X GET "http://localhost:8000/download/tts_c0490f9f_1765715910.mp3" \
  -o output.mp3

# 健康检查
curl -X GET "http://localhost:8000/health"
```

### JavaScript 客户端

```javascript
// 基础语音合成
async function synthesizeSpeech(text, loraName = null) {
    const response = await fetch('http://localhost:8000/synthesize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            lora_name: loraName,
            cfg_scale: 2.0,
            steps: 10
        })
    });

    const result = await response.json();
    if (result.status === 'success') {
        // 下载音频
        window.open(`http://localhost:8000/download/${result.audio_path.split('/').pop()}`);
    }
    return result;
}

// 使用示例
synthesizeSpeech("Hello, world!");
synthesizeSpeech("你好，世界！", "chinese_lora_model");
```

## 参数说明

### 主要参数

- **text** (必需): 要合成的文本内容
- **lora_name** (可选): LoRA 模型名称，可在 `/loras` 端点查看可用模型
- **cfg_scale** (可选): CFG 引导系数，默认 2.0，值越大越贴近提示
- **steps** (可选): 推理步数，默认 10，值越高质量越好但速度越慢
- **seed** (可选): 随机种子，-1 为随机，固定值可复现结果

### 声音克隆参数

- **ref_audio_path** (可选): 参考音频文件路径
- **ref_text** (可选): 参考音频对应的文本内容

## 文件结构

```
VoxCPM/
├── api_server.py           # API 服务器主文件 (完整功能)
├── start_api.py           # 启动脚本 (支持多种配置选项)
├── requirements_api.txt    # API 相关依赖包
├── test_async_api.py      # 完整异步测试脚本
├── test_network.py        # 网络配置测试工具
├── example_usage.py       # 基础使用示例
├── README_API.md         # API 详细文档
├── lora/                 # LoRA 模型目录
│   ├── lora1/            # 示例LoRA模型
│   │   ├── lora_weights.safetensors
│   │   └── lora_config.json
│   └── [其他LoRA模型]/
└── api_outputs/          # 生成的音频文件输出目录
    ├── tts_[task_id]_[timestamp].mp3
    └── ...
```

## 注意事项

1. **LoRA 模型路径**: LoRA 模型应放在 `lora/` 目录下，每个模型文件夹需包含 `lora_weights.safetensors` 文件
2. **音频格式**: API 自动输出 MP3 格式音频文件
3. **异步处理**:
   - 默认使用异步模式，避免长文本处理时的HTTP超时
   - 异步任务立即返回task_id，需要轮询状态获取结果
   - 同步模式适用于短文本，会直接等待结果
4. **模型加载**: 首次调用时需要加载模型，可能需要较长时间（1-3分钟）
5. **并发限制**: 后台默认同时处理1个任务，确保资源合理使用
6. **文件清理**: 建议定期调用 `/cleanup` 端点清理旧文件
7. **错误处理**: 所有错误都会返回详细的错误信息

## 性能优化建议

1. **批量处理**: 对于大量请求，建议使用异步客户端进行批量处理
2. **模型预热**: 首次请求前可以先用短文本预热模型
3. **内存管理**: 监控服务器内存使用，必要时重启服务
4. **文件存储**: 考虑使用对象存储服务替代本地文件存储

## 故障排除

### 常见问题

1. **模型加载失败**: 检查预训练模型路径是否正确
2. **LoRA 模型未找到**: 确保 LoRA 模型文件存在且路径正确
3. **音频生成失败**: 检查文本内容是否过长，尝试减少文本长度或增加内存
4. **连接超时**: 增加客户端的超时时间，首次模型加载可能需要较长时间

### 日志查看

服务器启动时会显示详细的日志信息，包括：
- 模型加载状态
- LoRA 配置信息
- 请求处理日志
- 错误详情

## 性能优化建议

1. **批量处理**: 对于大量请求，建议使用异步客户端进行批量处理
2. **模型预热**: 启动后先进行一次请求预热模型
3. **内存管理**: 监控服务器内存使用，必要时重启服务
4. **文件存储**: 考虑使用对象存储服务替代本地文件存储