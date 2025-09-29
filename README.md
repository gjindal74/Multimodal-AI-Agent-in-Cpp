# Real-Time Multimodal AI Agent in Pure C++

A production-grade, low-latency AI agent that sees, listens, thinks, and acts entirely in C++ with zero Python dependencies.

## What Makes This Jaw-Dropping?
This isn't your typical "call an API and hope for the best" AI project. This is a fully self-contained, hardware-accelerated, real-time multimodal system that runs entirely on local compute:

- **Real-Time Computer Vision:** YOLOv8 object detection at 30 FPS with ONNX Runtime optimization
- **Offline Speech Recognition:** Whisper.cpp integration for privacy-preserving STT (no cloud required)
- **Intelligent Reasoning:** LLM-powered decision making with context awareness
- **Low-Latency Pipeline:** Sub-100ms vision inference, 2-3s end-to-end voice processing
- **Production-Ready Architecture:** Thread-safe, modular design with proper memory management
- **Zero Python Dependency:** Pure C++ implementation for maximum performance and portability

## System Architecture
<img width="458" height="391" alt="Screenshot 2025-09-29 at 12 48 00 PM" src="https://github.com/user-attachments/assets/d0dedcbc-a8f8-4380-b0bf-9f24b7ecf13f" />

## Technical Deep Dive
 ### Vision Module

- **Model:** YOLOv8n ONNX (6.2MB, 80 COCO classes)
- **Optimization:**

Hardware-accelerated inference via ONNX Runtime
Custom NMS implementation with class-specific IoU thresholds
Temporal smoothing with Kalman-inspired tracking


- **Performance:** 25-30 FPS on MacBook Pro M1, 15-20 FPS on CPU
- **Key Innovation:** Dynamic confidence thresholds per object class (person: 0.5, small objects: 0.2)

 ### Audio Module

- **Engine:** Whisper.cpp (base.en model, 150MB)
- **Features:**

Real-time Voice Activity Detection (VAD) with adaptive thresholding
Circular buffer management for continuous audio streaming
Multi-threaded PortAudio integration for non-blocking capture
Dynamic gain control (100x amplification with clipping protection)


- **Latency:** 2-3s end-to-end (including 2s silence detection)
- **Key Innovation:** Energy-based VAD with dynamic threshold adaptation (10% of peak speech energy)

### LLM Reasoning Module

- **Planned:** llama.cpp integration for local inference
- **Context Window:** Combined vision detections + audio transcript → JSON action schema

### Action Executor

Safe command execution with whitelist validation
OS-level integration (macOS/Linux)
Planned actions: URL opening, media control, file operations, notifications

## Installation

```bash
# macOS
brew install cmake opencv portaudio pkg-config

# Linux (Ubuntu/Debian)
sudo apt-get install cmake libopencv-dev portaudio19-dev pkg-config

# Clone repository
git clone https://github.com/gjindal74/Multimodal-AI-Agent-in-Cpp.git
cd multimodal-agent-cpp

# Setup dependencies
mkdir -p third_party && cd third_party

# Build Whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp && make -j8 && cd ../..

# Download models
mkdir -p models
cd models

# YOLOv8n ONNX model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# Whisper base.en model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

cd ..

# Build project
mkdir build && cd build
cmake ..
make -j8

# Run
./src/agent_app
```

## Example Workflow

1. Launch: App initializes with live camera feed
2. Vision: Real-time object detection with bounding boxes
3. Voice Command: Say "What do you see?"
4. Transcription: Whisper processes → displays transcript
5. Reasoning: LLM analyzes scene + command → generates response
6. Action: Executes appropriate system command
