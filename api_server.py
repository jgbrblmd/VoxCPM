import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, List
import shutil
import uuid

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging

# Default pretrained model path relative to this repo
default_pretrained_path = str(project_root / "models" / "openbmb__VoxCPM1.5")

# Global variables
current_model: Optional[VoxCPM] = None
current_lora_path: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    lora_name: Optional[str] = None
    cfg_scale: float = 2.0
    steps: int = 10
    seed: int = -1
    # Optional voice cloning parameters
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None

class TTSResponse(BaseModel):
    task_id: str
    status: str
    message: str
    audio_path: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="VoxCPM LoRA TTS API",
    description="RESTful API for text-to-speech with LoRA model support",
    version="1.0.0"
)

def get_default_lora_config():
    """Return default LoRA config for hot-swapping support."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from lora_config.json if available."""
    lora_config_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(lora_config_file):
        try:
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
        except Exception as e:
            logger.warning(f"Failed to load lora_config.json: {e}")
    return None, None

def scan_lora_checkpoints(root_dir="lora"):
    """Scan for LoRA checkpoints in the lora directory."""
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            rel_path = os.path.relpath(root, root_dir)
            checkpoints.append(rel_path)

    return sorted(checkpoints, reverse=True)

def load_model_with_lora(lora_name: Optional[str] = None):
    """Load model with specified LoRA."""
    global current_model, current_lora_path

    base_model_path = default_pretrained_path

    # If LoRA is specified, try to get base model from its config
    if lora_name and lora_name != "None":
        full_lora_path = os.path.join("lora", lora_name)
        if os.path.exists(full_lora_path):
            lora_config_file = os.path.join(full_lora_path, "lora_config.json")

            if os.path.exists(lora_config_file):
                try:
                    with open(lora_config_file, "r", encoding="utf-8") as f:
                        lora_info = json.load(f)
                    saved_base_model = lora_info.get("base_model")

                    if saved_base_model and os.path.exists(saved_base_model):
                        base_model_path = saved_base_model
                        logger.info(f"Using base model from LoRA config: {base_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to read base_model from LoRA config: {e}")

    # Load model if not loaded or if LoRA changed
    if (current_model is None or
        (lora_name != current_lora_path and current_lora_path is not None)):

        logger.info(f"Loading base model: {base_model_path}")

        lora_config = None
        lora_weights_path = None

        if lora_name and lora_name != "None":
            full_lora_path = os.path.join("lora", lora_name)
            if os.path.exists(full_lora_path):
                lora_weights_path = full_lora_path
                lora_config, _ = load_lora_config_from_checkpoint(full_lora_path)
                if lora_config:
                    logger.info(f"Loaded LoRA config from {full_lora_path}/lora_config.json")
                else:
                    lora_config = get_default_lora_config()
                    logger.info("Using default LoRA config")

        if lora_config is None:
            lora_config = get_default_lora_config()

        current_model = VoxCPM.from_pretrained(
            hf_model_id=base_model_path,
            load_denoiser=False,
            optimize=False,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
        )

        current_lora_path = lora_name
        logger.info("Model loaded successfully")

    # Handle LoRA hot-swapping if model already loaded
    elif lora_name and lora_name != "None" and lora_name != current_lora_path:
        full_lora_path = os.path.join("lora", lora_name)
        logger.info(f"Hot-loading LoRA: {full_lora_path}")
        try:
            current_model.load_lora(full_lora_path)
            current_model.set_lora_enabled(True)
            current_lora_path = lora_name
        except Exception as e:
            logger.error(f"Error loading LoRA: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading LoRA: {e}")

    # Disable LoRA if None specified
    elif lora_name == "None" or lora_name is None:
        if current_model:
            current_model.set_lora_enabled(False)
            current_lora_path = None

def convert_to_mp3(audio_data, sample_rate, output_path):
    """Convert numpy audio data to MP3 format."""
    # First save as WAV
    temp_wav_path = output_path.replace('.mp3', '_temp.wav')
    sf.write(temp_wav_path, audio_data, sample_rate)

    # Convert to MP3
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(output_path, format="mp3", bitrate="128k")

    # Clean up temp file
    os.remove(temp_wav_path)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VoxCPM LoRA TTS API",
        "version": "1.0.0",
        "endpoints": {
            "synthesize": "/synthesize",
            "list_loras": "/loras",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": current_model is not None}

@app.get("/loras")
async def list_loras():
    """List available LoRA models."""
    try:
        loras = scan_lora_checkpoints()
        return {"loras": loras, "count": len(loras)}
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text with optional LoRA model."""
    try:
        # Validate LoRA name if provided
        if request.lora_name and request.lora_name != "None":
            lora_path = os.path.join("lora", request.lora_name)
            if not os.path.exists(lora_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"LoRA model '{request.lora_name}' not found"
                )

        # Load model with specified LoRA
        load_model_with_lora(request.lora_name)

        # Set seed if provided
        if request.seed != -1:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)

        # Handle reference audio if provided
        final_prompt_wav = None
        final_prompt_text = None

        if request.ref_audio_path and request.ref_audio_path.strip():
            if not os.path.exists(request.ref_audio_path):
                raise HTTPException(
                    status_code=404,
                    detail="Reference audio file not found"
                )

            final_prompt_wav = request.ref_audio_path

            # If no reference text provided, we'll proceed without it
            # Note: In a production environment, you might want to add ASR here
            if request.ref_text and request.ref_text.strip():
                final_prompt_text = request.ref_text.strip()

        # Generate audio
        logger.info(f"Generating audio for text: {request.text[:50]}...")
        audio_np = current_model.generate(
            text=request.text,
            prompt_wav_path=final_prompt_wav,
            prompt_text=final_prompt_text,
            cfg_value=request.cfg_scale,
            inference_timesteps=request.steps,
            denoise=False
        )

        # Create output directory
        output_dir = "api_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        task_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        filename = f"tts_{task_id}_{timestamp}.mp3"
        output_path = os.path.join(output_dir, filename)

        # Convert to MP3
        convert_to_mp3(audio_np, current_model.tts_model.sample_rate, output_path)

        logger.info(f"Audio generated successfully: {output_path}")

        return {
            "task_id": task_id,
            "status": "success",
            "message": "Speech synthesized successfully",
            "audio_path": output_path,
            "sample_rate": current_model.tts_model.sample_rate
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

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

@app.delete("/cleanup")
async def cleanup_old_files():
    """Clean up old generated files (older than 1 hour)."""
    try:
        output_dir = "api_outputs"
        if not os.path.exists(output_dir):
            return {"message": "No files to clean", "deleted_count": 0}

        current_time = time.time()
        deleted_count = 0

        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                # Check if file is older than 1 hour
                if current_time - os.path.getmtime(file_path) > 3600:
                    os.remove(file_path)
                    deleted_count += 1

        return {
            "message": "Cleanup completed",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Ensure output directory exists
    os.makedirs("api_outputs", exist_ok=True)

    # Run the API server on all network interfaces
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        reload=False,    # Disable reload for production
        log_level="info"
    )