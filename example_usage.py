"""
Example usage of the VoxCPM LoRA TTS API

This script demonstrates how to use the RESTful API to synthesize speech
with different LoRA models.
"""

import requests
import json
import time

# API base URL - 根据实际情况修改
BASE_URL = "http://localhost:8000"  # 本地访问
# BASE_URL = "http://192.168.1.100:8000"  # 局域网访问 (请替换为实际IP)

def test_api_connection():
    """Test if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on localhost:8000")
        return False

def list_available_loras():
    """List all available LoRA models."""
    try:
        response = requests.get(f"{BASE_URL}/loras")
        if response.status_code == 200:
            data = response.json()
            loras = data.get("loras", [])
            print(f"\n📋 Available LoRA models ({len(loras)}):")
            for i, lora in enumerate(loras, 1):
                print(f"  {i}. {lora}")
            return loras
        else:
            print(f"❌ Failed to list LoRAs: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error listing LoRAs: {e}")
        return []

def synthesize_speech(text, lora_name=None, **kwargs):
    """Synthesize speech using the API."""
    url = f"{BASE_URL}/synthesize"

    payload = {
        "text": text,
        "lora_name": lora_name,
        **kwargs
    }

    print(f"\n🎵 Synthesizing speech...")
    print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"   LoRA: {lora_name or 'None'}")

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Synthesis successful!")
            print(f"   Task ID: {data['task_id']}")
            print(f"   Audio file: {data['audio_path']}")
            print(f"   Sample rate: {data['sample_rate']} Hz")
            return data
        else:
            error_detail = response.json().get("detail", "Unknown error")
            print(f"❌ Synthesis failed: {error_detail}")
            return None
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        return None

def download_audio(filename, save_path=None):
    """Download generated audio file."""
    url = f"{BASE_URL}/download/{filename}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            if save_path is None:
                save_path = f"downloaded_{filename}"

            with open(save_path, 'wb') as f:
                f.write(response.content)

            print(f"✅ Audio downloaded to: {save_path}")
            return save_path
        else:
            print(f"❌ Failed to download audio: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return None

def main():
    """Main example function."""
    print("🚀 VoxCPM LoRA TTS API Example")
    print("=" * 40)

    # Test API connection
    if not test_api_connection():
        return

    # List available LoRAs
    loras = list_available_loras()

    # Example 1: Basic synthesis without LoRA
    print("\n" + "="*40)
    print("📝 Example 1: Basic TTS (no LoRA)")
    print("="*40)

    result1 = synthesize_speech(
        text="Hello, this is a test of the VoxCPM text-to-speech system without any LoRA model.",
        cfg_scale=2.0,
        steps=10
    )

    if result1:
        filename = os.path.basename(result1['audio_path'])
        download_audio(filename, "example_basic.mp3")

    # Example 2: TTS with LoRA (if available)
    if loras:
        print("\n" + "="*40)
        print("🎭 Example 2: TTS with LoRA model")
        print("="*40)

        # Use the first available LoRA
        selected_lora = loras[0]

        result2 = synthesize_speech(
            text="This is synthesized using a LoRA fine-tuned model for a specific voice style.",
            lora_name=selected_lora,
            cfg_scale=2.5,
            steps=15,
            seed=42
        )

        if result2:
            filename = os.path.basename(result2['audio_path'])
            download_audio(filename, f"example_lora_{selected_lora}.mp3")

    # Example 3: Voice cloning (if you have a reference audio)
    print("\n" + "="*40)
    print("🎤 Example 3: Voice cloning (optional)")
    print("="*40)

    # Uncomment and modify this if you have a reference audio file
    # result3 = synthesize_speech(
    #     text="This is a voice cloning example using the reference audio.",
    #     ref_audio_path="path/to/reference.wav",
    #     ref_text="This is the reference text for voice cloning.",
    #     cfg_scale=2.0,
    #     steps=20
    # )

    print("💡 To use voice cloning, uncomment the code above and provide:")
    print("   - ref_audio_path: Path to reference audio file")
    print("   - ref_text: Text content of the reference audio")

    print("\n✨ Example completed!")
    print(f"📁 Generated files are saved in the current directory")

if __name__ == "__main__":
    import os
    main()