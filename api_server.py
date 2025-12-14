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
import uvicorn

# Global variables - minimal for now
task_queue = Queue()
task_status: Dict[str, Dict] = {}
task_lock = threading.Lock()
worker_thread = None
current_model = None
current_lora_path = None
model_lock = threading.Lock()

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

# Initialize FastAPI app with minimal configuration
app = FastAPI(
    title="VoxCPM LoRA TTS API (Final)",
    description="Final version with real model support",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize background worker during app startup."""
    global worker_thread

    # Start background worker thread
    worker_thread = threading.Thread(target=task_worker, daemon=True)
    worker_thread.start()
    logger.info("Background task worker started")

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
    with model_lock:
        return {"status": "healthy", "model_loaded": current_model is not None}

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
            # For sync mode, process immediately
            result = process_task_real(task_id, request.model_dump())

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
            "max_concurrent": 1
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

def load_model_with_lora(lora_name: Optional[str] = None):
    """Load model with specified LoRA."""
    global current_model, current_lora_path

    with model_lock:
        # Check if model already loaded with correct LoRA
        if (current_model is not None and
            lora_name == current_lora_path):
            logger.info("Model already loaded with correct LoRA")
            return

        logger.info("Loading VoxCPM model...")

        try:
            # Import here to avoid loading during startup
            from voxcpm.core import VoxCPM
            from voxcpm.model.voxcpm import LoRAConfig
            import numpy as np
            import torch
            import soundfile as sf
            from pydub import AudioSegment

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

            # Load model
            logger.info(f"Loading model: {model_path}")
            current_model = VoxCPM.from_pretrained(
                hf_model_id=model_path,
                load_denoiser=False,
                optimize=False,
                lora_config=lora_config,
                lora_weights_path=lora_weights_path,
            )

            current_lora_path = lora_name
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            current_model = None
            current_lora_path = None
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

def process_task_real(task_id: str, request_data: Dict):
    """Process task with real model."""
    try:
        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.0, message="开始处理任务...")

        # Load model with specified LoRA
        update_task_status_safe(task_id, TaskStatus.PROCESSING, progress=0.1, message="加载模型...")
        lora_name = request_data.get('lora_name')
        load_model_with_lora(lora_name)

        if not current_model:
            raise Exception("Failed to load model")

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

        # Generate audio
        with model_lock:  # Ensure thread safety during generation
            audio_np = current_model.generate(
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
        convert_to_mp3(audio_np, current_model.tts_model.sample_rate, output_path)

        update_task_status_safe(task_id, TaskStatus.COMPLETED, progress=1.0,
                             message="语音合成完成", audio_path=output_path)
        logger.info(f"Task {task_id} completed successfully")

        return {"status": "completed", "audio_path": output_path, "message": "语音合成完成"}

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

def task_worker():
    """Background worker to process tasks from queue."""
    logger.info("Task worker thread started")
    while True:
        try:
            task_id, request_data = task_queue.get(timeout=2.0)
            if task_id:
                logger.info(f"Processing task {task_id}")
                process_task_real(task_id, request_data)
        except Exception:
            continue

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("api_outputs", exist_ok=True)

    logger.info("Starting VoxCPM API server with real model support")

    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )