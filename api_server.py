import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict
import uuid
import threading
from queue import Queue
from datetime import datetime
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import FastAPI components
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Global variables - true multi-GPU approach
task_queue = Queue()
task_status: Dict[str, Dict] = {}
task_lock = threading.Lock()
worker_threads = []  # Multiple worker threads
current_models = {}  # Multiple model instances (GPU -> model)
current_lora_paths = {}  # Track LoRA for each GPU
model_locks = {}  # Separate lock for each GPU
available_gpus = []  # List of available GPU devices
max_concurrent_tasks = 2  # Number of GPUs for parallel processing

# Task status constants
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    lora_name: Optional[str] = None
    cfg_scale: float = 2.0
    steps: int = 10
    seed: int = -1
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    async_mode: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global worker_threads, max_concurrent_tasks, available_gpus, model_locks

    # Startup
    logger.info("Starting VoxCPM API server...")

    # Detect available GPUs
    available_gpus = detect_available_gpus()
    max_concurrent_tasks = len(available_gpus)

    # Initialize locks for each GPU
    for gpu_id in available_gpus:
        model_locks[gpu_id] = threading.Lock()

    logger.info(f"Detected {len(available_gpus)} GPUs: {available_gpus}")
    logger.info(f"Starting {max_concurrent_tasks} worker threads for true parallel processing")

    # Start multiple worker threads, each bound to a specific GPU
    for i in range(max_concurrent_tasks):
        gpu_id = available_gpus[i]
        worker_thread = threading.Thread(target=task_worker, args=(i, gpu_id), daemon=True)
        worker_thread.start()
        worker_threads.append(worker_thread)
        logger.info(f"Background task worker {i} started for GPU {gpu_id}")

    yield

    # Shutdown
    logger.info("Shutting down VoxCPM API server...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="VoxCPM LoRA TTS API (Multi-GPU)",
    description="VoxCPM TTS API with multi-GPU support",
    version="1.0.0",
    lifespan=lifespan
)

def detect_available_gpus():
    """Detect available GPU devices."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            # For AMD GPUs, check for ROCm devices
            try:
                import subprocess
                result = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse ROCm output to get GPU IDs
                    gpu_ids = []
                    for line in result.stdout.split('\n'):
                        if 'GPU' in line and 'Device' in line:
                            # Extract GPU ID from ROCm output
                            parts = line.split()
                            for part in parts:
                                if part.isdigit():
                                    gpu_ids.append(int(part))
                                    break
                    return gpu_ids if gpu_ids else [0, 1]  # Default to 2 GPUs for AMD
            except:
                pass

            # Default fallback
            logger.warning("Could not detect GPUs, defaulting to 2 workers")
            return [0, 1]

    except ImportError:
        logger.warning("PyTorch not available, defaulting to 2 workers")
        return [0, 1]

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "VoxCPM LoRA TTS API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    loaded_gpus = []
    for gpu_id in available_gpus:
        if gpu_id in current_models and current_models[gpu_id] is not None:
            loaded_gpus.append(gpu_id)

    return {
        "status": "healthy",
        "model_loaded": len(loaded_gpus) > 0,  # Keep for backward compatibility
        "available_gpus": available_gpus,
        "loaded_gpus": loaded_gpus,
        "max_concurrent_tasks": max_concurrent_tasks
    }

@app.get("/loras")
async def list_loras():
    """List available LoRA models."""
    try:
        lora_dir = "lora"
        if not os.path.exists(lora_dir):
            return {"loras": [], "count": 0}

        checkpoints = []
        for root, dirs, files in os.walk(lora_dir):
            if "lora_weights.safetensors" in files:
                rel_path = os.path.relpath(root, lora_dir)
                checkpoints.append(rel_path)

        return {"loras": sorted(checkpoints, reverse=True), "count": len(checkpoints)}
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return {"loras": [], "count": 0, "error": str(e)}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech endpoint with real model support."""
    try:
        # Validate LoRA name if provided
        if request.lora_name and request.lora_name != "None":
            lora_path = os.path.join("lora", request.lora_name)
            if not os.path.exists(lora_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"LoRA model '{request.lora_name}' not found"
                )

        # Generate task ID
        task_id = str(uuid.uuid4())[:8]
        estimated_time = max(len(request.text) * 0.3 + request.steps, 30)

        # Initialize task status
        with task_lock:
            task_status[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "message": "任务已提交，等待处理...",
                "progress": 0.0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "estimated_time": estimated_time
            }

        if request.async_mode:
            # Add to queue without blocking
            task_queue.put((task_id, request.model_dump()))
            logger.info(f"Task {task_id} queued for async processing")

            return {
                "task_id": task_id,
                "status": "submitted",
                "message": "任务已提交，请使用task_id查询处理状态",
                "estimated_time": estimated_time,
                "progress": 0.0
            }
        else:
            # For sync mode, process immediately on first available GPU
            gpu_id = available_gpus[0] if available_gpus else 0
            result = process_task_real(task_id, request.model_dump(), 0, gpu_id)

            return {
                "task_id": task_id,
                "status": result.get("status", "unknown"),
                "message": result.get("message", ""),
                "audio_path": result.get("audio_path"),
                "progress": result.get("progress", 0)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    with task_lock:
        if task_id not in task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task_info = task_status[task_id].copy()
        return task_info

@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 50):
    """List all tasks."""
    with task_lock:
        tasks = list(task_status.values())
        if status:
            tasks = [task for task in tasks if task["status"] == status]
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        tasks = tasks[:limit]

        processing_count = sum(1 for task in task_status.values() if task["status"] == "processing")

        return {
            "tasks": tasks,
            "total": len(tasks),
            "processing": processing_count,
            "max_concurrent": max_concurrent_tasks,
            "available_gpus": len(available_gpus)
        }

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio file."""
    file_path = os.path.join("api_outputs", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )

def load_model_with_lora(lora_name: Optional[str] = None, gpu_id: int = 0):
    """Load model with specified LoRA on specific GPU."""
    global current_models, current_lora_paths

    with model_locks[gpu_id]:
        # Check if model already loaded with correct LoRA on this GPU
        if (gpu_id in current_models and current_models[gpu_id] is not None and
            lora_name == current_lora_paths.get(gpu_id)):
            logger.info(f"Model already loaded on GPU {gpu_id} with correct LoRA")
            return

        logger.info(f"Loading VoxCPM model on GPU {gpu_id}...")

        try:
            # Import here to avoid loading during startup
            from voxcpm.core import VoxCPM
            from voxcpm.model.voxcpm import LoRAConfig
            import numpy as np
            import torch
            import soundfile as sf
            from pydub import AudioSegment

            # Set CUDA device before loading
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                # Set environment variable for this thread
                old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Default model path
            model_path = str(project_root / "models" / "openbmb__VoxCPM1.5")

            # Load LoRA config if available
            lora_config = get_default_lora_config()
            lora_weights_path = None

            if lora_name and lora_name != "None":
                lora_path = os.path.join("lora", lora_name)
                if os.path.exists(lora_path):
                    lora_weights_path = lora_path
                    loaded_config, _ = load_lora_config_from_checkpoint(lora_path)
                    if loaded_config:
                        lora_config = loaded_config

            # Load model - it should automatically use the set CUDA device
            logger.info(f"Loading model: {model_path} for GPU {gpu_id}")
            model = VoxCPM.from_pretrained(
                hf_model_id=model_path,
                load_denoiser=False,
                optimize=False,
                lora_config=lora_config,
                lora_weights_path=lora_weights_path,
            )

            # Restore CUDA_VISIBLE_DEVICES
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

            current_models[gpu_id] = model
            current_lora_paths[gpu_id] = lora_name
            logger.info(f"Model loaded successfully on GPU {gpu_id}")

        except Exception as e:
            logger.error(f"Error loading model on GPU {gpu_id}: {e}")
            if gpu_id in current_models:
                current_models[gpu_id] = None
            if gpu_id in current_lora_paths:
                current_lora_paths[gpu_id] = None
            raise

def get_default_lora_config():
    """Return default LoRA config."""
    try:
        from voxcpm.model.voxcpm import LoRAConfig
        return LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            r=32,
            alpha=16,
            target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
            target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
    except Exception:
        return None

def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from checkpoint."""
    try:
        lora_config_file = os.path.join(lora_path, "lora_config.json")
        if os.path.exists(lora_config_file):
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                from voxcpm.model.voxcpm import LoRAConfig
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
    except Exception:
        pass
    return None, None

def convert_to_mp3(audio_data, sample_rate, output_path):
    """Convert numpy audio data to MP3 format."""
    try:
        import soundfile as sf
        from pydub import AudioSegment

        # Save as WAV first
        temp_wav_path = output_path.replace('.mp3', '_temp.wav')
        sf.write(temp_wav_path, audio_data, sample_rate)

        # Convert to MP3
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(output_path, format="mp3", bitrate="128k")

        # Clean up temp file
        os.remove(temp_wav_path)
    except Exception as e:
        logger.error(f"Error converting to MP3: {e}")

def process_task_real(task_id: str, request_data: Dict, worker_id: int = 0, gpu_id: int = 0):
    """Process task with real model on specific GPU."""
    try:
        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.0,
                             message=f"开始在GPU {gpu_id}处理任务...")

        # Load model with specified LoRA on this GPU
        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.1, message="加载模型...")
        lora_name = request_data.get('lora_name')
        load_model_with_lora(lora_name, gpu_id)

        if gpu_id not in current_models or current_models[gpu_id] is None:
            raise Exception(f"Failed to load model on GPU {gpu_id}")

        model = current_models[gpu_id]

        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.2, message="准备生成参数...")

        # Set seed if provided
        seed = request_data.get('seed', -1)
        if seed != -1:
            import torch
            import numpy as np
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Handle reference audio
        ref_audio_path = request_data.get('ref_audio_path')
        ref_text = request_data.get('ref_text')

        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.3, message="生成音频中...")

        # Generate audio on specific GPU with lock
        with model_locks[gpu_id]:
            audio_np = model.generate(
                text=request_data['text'],
                prompt_wav_path=ref_audio_path,
                prompt_text=ref_text,
                cfg_value=request_data.get('cfg_scale', 2.0),
                inference_timesteps=request_data.get('steps', 10),
                denoise=False
            )

        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.8, message="保存音频文件...")

        # Create output directory
        output_dir = "api_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        timestamp = int(time.time())
        filename = f"tts_{task_id}_{timestamp}.mp3"
        output_path = os.path.join(output_dir, filename)

        # Convert to MP3
        convert_to_mp3(audio_np, model.tts_model.sample_rate, output_path)

        update_task_status_safe(task_id, TaskStatus.COMPLETED, progress=1.0,
                             message=f"语音合成完成 (GPU {gpu_id})", audio_path=output_path)
        logger.info(f"Task {task_id} completed successfully on GPU {gpu_id}")

        return {"status": "completed", "audio_path": output_path, "message": f"语音合成完成"}

    except Exception as e:
        error_msg = f"Task failed: {str(e)}"
        update_task_status_safe(task_id, TaskStatus.FAILED, message=error_msg, error=str(e))
        logger.error(f"Task {task_id} failed: {e}")
        return {"status": "failed", "error": str(e)}

def update_task_status_safe(task_id: str, status: str, progress: float = None,
                           message: str = None, audio_path: str = None,
                           error: str = None):
    """Update task status safely."""
    try:
        with task_lock:
            if task_id in task_status:
                task_status[task_id]['status'] = status
                task_status[task_id]['updated_at'] = datetime.now().isoformat()

                if progress is not None:
                    task_status[task_id]['progress'] = progress
                if message:
                    task_status[task_id]['message'] = message
                if audio_path:
                    task_status[task_id]['audio_path'] = audio_path
                if error:
                    task_status[task_id]['error'] = error
    except Exception as e:
        logger.error(f"Error updating task status: {e}")

def task_worker(worker_id: int, gpu_id: int):
    """Background worker to process tasks from queue on specific GPU."""
    logger.info(f"Task worker {worker_id} started for GPU {gpu_id}")

    while True:
        try:
            task_id, request_data = task_queue.get(timeout=2.0)
            if task_id:
                logger.info(f"Worker {worker_id} (GPU {gpu_id}) processing task {task_id}")
                process_task_real(task_id, request_data, worker_id, gpu_id)
        except Exception:
            continue

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("api_outputs", exist_ok=True)

    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )