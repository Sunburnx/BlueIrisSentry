# BlueIrisSentry

> A high-performance AI gateway that integrates YOLOv8 with Blue Iris for real-time object detection in IP camera surveillance systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-zone)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OIV7-orange.svg)](https://github.com/ultralytics/ultralytics)

## Overview

**BlueIrisSentry** is a production-ready, GPU-accelerated object detection server that seamlessly integrates YOLOv8 (Open Images V7 dataset) with Blue Iris IP camera software. It serves as a drop-in replacement for CodeProject.AI and DeepStack, delivering state-of-the-art (SOTA) object detection with **600 classes** and **sub-100ms inference times**.

### Key Features

- **600 Object Classes**: YOLOv8x trained on Open Images V7 (OIV7) dataset
- **Lightning Fast**: 50-115ms inference on NVIDIA RTX 6000 Pro Blackwell
- **CodeProject.AI Compatible**: Drop-in replacement with full Blue Iris integration
- **GPU Accelerated**: Optimized for CUDA 13.0 / 12.9.1 with NVIDIA Container Runtime
- **Production Ready**: Docker containerized with systemd auto-start
- **Real-time Monitoring**: Built-in web UI for testing and debugging
- **Comprehensive Detection**: People, vehicles, animals, objects, and more
- **Flexible Configuration**: Adjustable confidence thresholds, image sizes, and model parameters

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Blue Iris Configuration](#blue-iris-configuration)
- [API Documentation](#api-documentation)
- [Performance Benchmarks](#performance-benchmarks)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Quick Start

### 1. Clone and Build

```bash
cd ~/containers/yolo-detector

# Build Docker image
docker compose build

# Start container
docker compose up -d

# Verify GPU access
docker exec yolo-detector nvidia-smi
```

### 2. Test the Server

```bash
# Health check
curl http://localhost:9090/health | jq .

# Test detection with sample image
curl -X POST http://localhost:9090/v1/vision/custom/yolov8x-oiv7 \
  -F "image=@test_images/person.jpg" \
  -F "min_confidence=0.25" | jq .

# Open Web UI
firefox http://192.168.1.3:9091/
```

### 3. Configure Blue Iris

1. Open Blue Iris Settings → **AI** tab
2. Enter server: `http://192.168.1.3:9090`
3. Click **Test** to verify connection
4. Configure cameras (see [Blue Iris Configuration](#blue-iris-configuration))

## System Requirements

### Hardware

| Component | Requirement | Notes |
|-----------|-------------|-------|
| GPU | NVIDIA GPU with CUDA support | Tested: RTX 6000 Pro Blackwell (96GB) |
| Driver | NVIDIA Driver 580+ | Open-source kernel module recommended for Blackwell |
| RAM | 8GB minimum | 16GB+ recommended for multiple streams |
| CPU | 4+ cores | AMD EPYC or Intel Xeon recommended |
| Storage | 10GB+ | For model weights and logs |

### Software

- **OS**: Ubuntu 24.04+ (tested on Ubuntu 25.04 Plucky Puffin)
- **Docker**: Docker Engine 20.10+ with Compose V2
- **NVIDIA Container Toolkit**: For GPU passthrough
- **Blue Iris**: Version 5.6.9.8+ (Windows)

### Network

- Local network connectivity between Blue Iris PC and detection server
- Open ports: **9090** (API), **9091** (Web UI)

## Installation

### Step 1: Install Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Clone Repository

```bash
mkdir -p ~/containers
cd ~/containers
git clone <repository-url> yolo-detector
cd yolo-detector
```

### Step 3: Build and Start Container

```bash
# Build Docker image (downloads CUDA base image and dependencies)
docker compose build

# Start container in detached mode
docker compose up -d

# Verify container is running
docker ps | grep yolo-detector

# Check logs for startup messages
docker logs -f yolo-detector
```

Expected log output:
```
INFO:     Model loaded: yolov8x-oiv7.pt
INFO:     Using device: cuda
INFO:     GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
INFO:     Started server on 0.0.0.0:9090
INFO:     Application startup complete.
```

### Step 4: Verify GPU Access

```bash
docker exec yolo-detector nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blackwell...  Off |   00000000:01:00.0 Off |                  Off |
| 30%   35C    P8             25W /  300W |    2048MiB /  97247MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Step 5: Install Systemd Service (Optional - for auto-start)

```bash
# Copy service file
sudo cp yolo-detector.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service for auto-start on boot
sudo systemctl enable yolo-detector.service

# Start service
sudo systemctl start yolo-detector.service

# Check status
sudo systemctl status yolo-detector.service
```

## Blue Iris Configuration

### Global AI Configuration

1. Open **Blue Iris** on Windows PC
2. Click **Settings** (gear icon) → **AI** tab
3. Configure AI server:
   - **Uncheck** "Use AI server on localhost"
   - **Server URL**: `http://192.168.1.3:9090` (replace with your server IP)
   - Leave **API Path** empty (BlueIrisSentry handles multiple endpoints)
4. Click **Test** button
   - Expected: "Server is online" or "Unknown server: is online"
5. Click **Start** to enable AI processing
6. **Status** section should show:
   - "Custom models list updated"
   - Models: yolov8x-oiv7

### Per-Camera Configuration

For each camera you want AI detection on:

#### 1. Motion Sensor Setup

1. Right-click camera → **Camera Properties**
2. Go to **Trigger** tab
3. Check **Motion Sensor**
4. Click **Configure** button next to Motion Sensor
5. Configure zones:
   - Highlight areas you want monitored (Zone A = green)
   - Leave areas you want ignored uncolored
6. Adjust sensitivity:
   - **Minimum object size**: 500-1000 pixels (adjust per camera)
   - **Minimum contrast**: 10-15
   - **Make time**: 0.1-0.2 seconds
7. Click **OK**

#### 2. Artificial Intelligence Setup

1. In **Trigger** tab, check **Artificial Intelligence**
2. Click **Artificial Intelligence** button to configure
3. AI Configuration settings:
   - **Check "Enabled"**
   - **Objects to detect**: `person,car,truck,van,motorcycle,bicycle,dog,cat,deer`
   - **Min confidence**: 40% (start here, adjust based on results)
   - **Real-time images**: 10-15 (more = slower but more accurate)
   - **Analyze one each**: 500ms (how often to analyze)
   - **Begin analysis with**: Motion leading images
   - **Uncheck** "Use main stream if available" (slower, not more accurate)
   - **AI server**: Should auto-select your configured server
   - **Model**: Select "yolov8x-oiv7" if available
4. Click **OK**

#### 3. Alert Configuration

1. Go to **Alerts** tab
2. Check **Add to the alerts list**
3. Check **JPEG files** (required for AI analysis)
4. Configure alert actions (email, push notification, recording, etc.)
5. In **To cancel** field: `nothing found:0`
   - This cancels alerts when AI detects no objects of interest
6. Click **OK**

### Testing

1. Walk in front of a configured camera
2. Open Blue Iris → **Status** → **AI** tab
3. You should see:
   - Detection requests appearing
   - Object labels with confidence percentages (e.g., "person:92%")
   - Inference times (50-115ms typical)
   - Green checkmark = alert confirmed
   - Red X = alert canceled (no objects found or below threshold)

## API Documentation

BlueIrisSentry implements multiple API endpoints for compatibility with Blue Iris and CodeProject.AI.

### Base URL

```
http://192.168.1.3:9090
```

Replace `192.168.1.3` with your server's IP address.

### Endpoints

#### 1. Custom Model Detection (Primary Blue Iris Endpoint)

**Endpoint**: `POST /v1/vision/custom/{model_name}`

This is the primary endpoint Blue Iris uses when configured with a specific model.

**Request**:
```bash
curl -X POST http://192.168.1.3:9090/v1/vision/custom/yolov8x-oiv7 \
  -F "image=@camera_snapshot.jpg" \
  -F "min_confidence=0.25"
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| image | File | Yes | - | JPEG/PNG image file |
| min_confidence | Float | No | 0.18 | Minimum confidence threshold (0.0-1.0) |

**Response** (Success):
```json
{
  "success": true,
  "predictions": [
    {
      "label": "person",
      "confidence": 0.92,
      "x_min": 100,
      "y_min": 50,
      "x_max": 300,
      "y_max": 400
    },
    {
      "label": "car",
      "confidence": 0.88,
      "x_min": 500,
      "y_min": 200,
      "x_max": 900,
      "y_max": 600
    }
  ],
  "count": 2,
  "inference_ms": 68.2,
  "process_ms": 72.5,
  "model": "yolov8x-oiv7.pt",
  "image_size": [1920, 1080],
  "timestamp": "2025-11-01T16:23:46.789Z"
}
```

**Response** (Error):
```json
{
  "success": false,
  "error": "No image file provided",
  "message": "Image is required",
  "code": "MISSING_IMAGE"
}
```

#### 2. Health Check

**Endpoint**: `GET /health`

Check server status and GPU availability.

**Request**:
```bash
curl http://192.168.1.3:9090/health | jq .
```

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
  "gpu_memory_total_mb": 97247,
  "gpu_memory_used_mb": 2048,
  "model_loaded": true,
  "model_name": "yolov8x-oiv7.pt",
  "uptime_seconds": 3600.0,
  "version": "1.0.0",
  "cuda_version": "12.9"
}
```

#### 3. Model Information

**Endpoint**: `GET /models/info`

Get detailed model statistics and configuration.

**Request**:
```bash
curl http://192.168.1.3:9090/models/info | jq .
```

**Response**:
```json
{
  "model_name": "yolov8x-oiv7",
  "model_path": "/app/models/yolov8x-oiv7.pt",
  "model_size_mb": 136.2,
  "num_classes": 600,
  "input_size": 1280,
  "device": "cuda:0",
  "half_precision": false,
  "batch_size": 1,
  "warmup_complete": true,
  "total_detections": 12847,
  "avg_inference_ms": 68.3
}
```

#### 4. List Available Classes

**Endpoint**: `GET /classes`

Get all 600 available detection classes.

**Request**:
```bash
curl http://192.168.1.3:9090/classes | jq .
```

**Response**:
```json
{
  "total": 600,
  "classes": {
    "0": "Accordion",
    "1": "Adhesive tape",
    "2": "Aircraft",
    ...
    "599": "Zucchini"
  },
  "categories": {
    "people": ["Person", "Man", "Woman", "Boy", "Girl", "Human face"],
    "vehicles": ["Car", "Truck", "Van", "Bus", "Motorcycle", "Bicycle"],
    "animals": ["Dog", "Cat", "Horse", "Cattle", "Pig", "Sheep", "Bird", "Bear", "Deer"]
  }
}
```

#### 5. CodeProject.AI Compatibility Endpoints

**Status Ping**: `GET /v1/status/ping` or `GET /v1/server/status/ping`

**List Models**: `GET /v1/vision/custom/list` or `POST /v1/vision/custom/list`

**Web UI**: `GET /` - Serves HTML dashboard for browser-based testing

### Common Object Classes for Security/Surveillance

**People (17+ classes)**:
- Person, Man, Woman, Boy, Girl, Human body, Human face, Human arm, Human hand, Human leg

**Vehicles (28+ classes)**:
- Car, Truck, Van, Bus, Taxi, Limousine, Ambulance, Motorcycle, Bicycle, Skateboard, Golf cart, Wheelchair

**Animals (40+ classes)**:
- Dog, Cat, Horse, Cattle, Pig, Sheep, Goat, Chicken, Deer, Fox, Bear, Raccoon, Skunk, Squirrel, Rabbit, Bird, Eagle, Owl, Duck, Insect, Spider

**Security-Relevant Objects (15+ classes)**:
- Backpack, Suitcase, Handbag, Briefcase, Luggage, Package, Box, Weapon, Knife, Ladder, Tool

**Full list**: See `/classes` endpoint or [Open Images V7 documentation](https://storage.googleapis.com/openimages/web/index.html)

## Performance Benchmarks

Tested on **NVIDIA RTX 6000 Pro Blackwell** (96GB GDDR7, CUDA 13.0):

### Inference Time Results

| Test Image | Resolution | Objects Detected | Inference Time | Total Processing |
|------------|------------|------------------|----------------|------------------|
| Person + Dog | 1920x1080 | 3 (person, dog, dog) | 68.2ms | 72.1ms |
| Carport Scene | 3840x2160 | 31 (insect, car, etc.) | 114.5ms | 116.8ms |
| Deer | 1920x1080 | 1 (deer) | 52.3ms | 55.1ms |
| Person on Bicycle | 3840x2160 | 12 (person, bicycle, various) | 87.4ms | 91.2ms |

**Average Inference Time**: 50-115ms
**Target**: <100ms ✅
**GPU Memory Usage**: ~2GB VRAM
**Concurrent Requests**: 10+ simultaneous without degradation

### Real-World Blue Iris Performance

| Camera | Resolution | FPS | Objects | Avg Inference | Notes |
|--------|------------|-----|---------|---------------|-------|
| PCI_4 (Carport) | 3840x2160 | 15 | Animals, Cars | 110ms | 4K camera, multiple objects |
| G4 (Front Door) | 1920x1080 | 20 | People, Vehicles | 65ms | 1080p camera, typical use |
| G4 (Backyard) | 1920x1080 | 20 | Animals, People | 58ms | 1080p camera, wildlife |
| G3 (Driveway) | 1280x720 | 30 | Vehicles | 45ms | 720p camera, single object type |

**Production Status**: ✅ Successfully detecting objects in real-time across multiple cameras

## Configuration

### Environment Variables

Edit `docker-compose.yml` to customize server behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `yolov8x-oiv7.pt` | YOLOv8 model filename |
| `CONFIDENCE_THRESHOLD` | `0.18` | Default minimum confidence |
| `DEFAULT_IMGSZ` | `1280` | Default image size for inference |
| `DEFAULT_IOU` | `0.45` | Intersection over Union threshold |
| `DEFAULT_MAX_DET` | `300` | Maximum detections per image |
| `API_PORT` | `9090` | API server port |
| `WEB_UI_PORT` | `9091` | Web UI port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_QUEUE_SIZE` | `10` | Maximum request queue size |
| `SAVE_DEBUG_IMAGES` | `false` | Save annotated images to output/ |

### Apply Configuration Changes

```bash
# Stop container
docker compose down

# Edit configuration
nano docker-compose.yml

# Rebuild and restart
docker compose build
docker compose up -d

# Verify changes
docker logs -f yolo-detector
```

### Custom Model

To use a different YOLOv8 model:

1. Download or train your model (`.pt` file)
2. Place it in the `models/` directory
3. Update `MODEL_NAME` environment variable in `docker-compose.yml`
4. Restart container: `docker compose restart`

Available pretrained models:
- `yolov8n.pt` - Nano (smallest, fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (80 COCO classes)
- `yolov8x-oiv7.pt` - Extra Large (600 Open Images V7 classes) **← Current**

## Troubleshooting

### Issue: Container Won't Start

**Symptoms**:
- `docker compose up -d` fails
- Container exits immediately after starting

**Diagnosis**:
```bash
# Check container status
docker ps -a | grep yolo-detector

# View logs
docker compose logs yolo-detector

# Check GPU devices
ls -l /dev/nvidia*
```

**Solutions**:

1. **GPU devices not found**:
```bash
# Verify NVIDIA driver is loaded
nvidia-smi

# Restart NVIDIA persistence daemon
sudo systemctl restart nvidia-persistenced
sleep 5

# Restart Docker
sudo systemctl restart docker
```

2. **CUDA version mismatch**:
```bash
# Verify CUDA version in container
docker run --rm --gpus all nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 nvcc --version

# If mismatch, rebuild with correct base image
docker compose build --no-cache
```

3. **Port already in use**:
```bash
# Check what's using port 9090
sudo lsof -i :9090

# Option 1: Stop the conflicting service
# Option 2: Change port in docker-compose.yml
```

### Issue: Blue Iris Can't Connect

**Symptoms**:
- Blue Iris shows "Connection refused" or timeout
- Test button in Blue Iris fails

**Diagnosis**:
```bash
# Test from server itself
curl http://localhost:9090/health

# Test from Blue Iris machine (run in Windows PowerShell)
curl http://192.168.1.3:9090/health

# Check if container is listening
docker exec yolo-detector netstat -tuln | grep 9090
```

**Solutions**:

1. **Firewall blocking**:
```bash
# Check firewall status
sudo ufw status

# Allow port 9090
sudo ufw allow 9090/tcp

# Or allow from specific IP (Blue Iris machine)
sudo ufw allow from 192.168.1.8 to any port 9090
```

2. **Wrong IP address**:
```bash
# Find correct IP address
ip addr show

# Update Blue Iris configuration with correct IP
```

3. **Container not listening on all interfaces**:
```bash
# Check server.py has host="0.0.0.0"
docker exec yolo-detector grep "host=" /app/server.py

# Should see: uvicorn.run(app, host="0.0.0.0", port=...)
```

### Issue: Slow Inference Times (>200ms)

**Symptoms**:
- Detection works but takes longer than expected
- Blue Iris shows inference times >200ms

**Diagnosis**:
```bash
# Check if GPU is being used
docker logs yolo-detector | grep "Using device"
# Should show: "Using device: cuda"

# Monitor GPU utilization during detection
docker exec yolo-detector nvidia-smi

# Check GPU memory usage
docker exec yolo-detector nvidia-smi --query-gpu=memory.used --format=csv
```

**Solutions**:

1. **CPU fallback instead of GPU**:
```bash
# Verify GPU is visible to container
docker exec yolo-detector nvidia-smi

# Restart container to reinitialize GPU
docker compose restart

# Check logs for GPU initialization
docker logs yolo-detector | head -20
```

2. **Image resolution too high**:
- Reduce `DEFAULT_IMGSZ` in docker-compose.yml (try 640 or 960)
- Blue Iris will send full resolution images, but YOLO will resize them

3. **Too many concurrent requests**:
```bash
# Check if requests are queued
docker logs yolo-detector | grep "queue"

# Reduce Blue Iris "Real-time images" setting to 5-10
```

### Issue: Model Download Fails

**Symptoms**:
- Container starts but model file is missing
- Error: "Model file not found"

**Solutions**:

1. **Manual download**:
```bash
cd ~/containers/yolo-detector/models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-oiv7.pt

# Verify download
ls -lh yolov8x-oiv7.pt

# Restart container
docker compose restart
```

2. **Disk space**:
```bash
# Check available space
df -h ~/containers/yolo-detector

# Clean up Docker if needed
docker system prune -a
```

### Issue: Wrong Detections or Too Many False Positives

**Symptoms**:
- Objects detected that aren't there
- Too many alerts for insignificant objects

**Solutions**:

1. **Increase confidence threshold**:
- Blue Iris: Set "Min confidence" to 50-60%
- Or increase `CONFIDENCE_THRESHOLD` in docker-compose.yml

2. **Refine object list**:
- In Blue Iris camera settings, only include objects you care about
- Example: `person,car,truck` instead of including animals if you only want vehicles/people

3. **Adjust motion sensor**:
- Increase "Minimum object size" to ignore small movements
- Configure zones to exclude problematic areas (trees, flags, etc.)

4. **Use Blue Iris static object filtering**:
- Blue Iris can ignore objects that don't move between frames
- This helps eliminate false positives from parked cars, etc.

### Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size or image size |
| `RuntimeError: CUDA error: no kernel image` | CUDA compute capability mismatch | Use correct CUDA base image (12.9.1) |
| `ModuleNotFoundError: ultralytics` | Dependencies not installed | Rebuild container: `docker compose build` |
| `OSError: [Errno 28] No space left` | Disk full | Clean up: `docker system prune -a` |
| `Connection refused` | Server not running or firewall | Check container status and firewall |
| `400 Bad Request` | Invalid image format | Check image file is valid JPEG/PNG |
| `404 Not Found` | Wrong endpoint | Verify Blue Iris URL configuration |

### Enable Debug Logging

For detailed troubleshooting:

1. Edit `docker-compose.yml`:
```yaml
environment:
  - LOG_LEVEL=DEBUG
  - SAVE_DEBUG_IMAGES=true
```

2. Restart container:
```bash
docker compose restart
```

3. View debug logs:
```bash
docker logs -f yolo-detector

# Debug images saved to:
ls -lh output/detections/
```

## Development

### Project Structure

```
~/containers/yolo-detector/
├── docker-compose.yml          # Docker Compose orchestration
├── Dockerfile                  # Container build instructions
├── .env                        # Environment variables (optional)
├── .gitignore                  # Git ignore rules
├── yolo-detector.service       # Systemd service file
├── README.md                   # This file
├── CLAUDE.md                   # Detailed implementation guide
│
├── app/                        # Application source code
│   ├── server.py               # FastAPI server (main entry point)
│   ├── detector.py             # YOLOv8 detection logic
│   ├── config.py               # Configuration management
│   ├── utils.py                # Helper functions
│   ├── web_ui.html             # Web-based testing interface
│   └── requirements.txt        # Python dependencies
│
├── models/                     # Model weights (auto-downloaded)
│   └── yolov8x-oiv7.pt         # YOLOv8x Open Images V7 model (~136MB)
│
├── config/                     # Configuration files
│   └── (empty - for future use)
│
├── test_images/                # Sample test images
│   ├── person.jpg
│   ├── car.jpg
│   └── person_dog.jpg
│
├── output/                     # Detection results (if SAVE_DEBUG_IMAGES=true)
│   └── detections/
│
└── logs/                       # Application logs
    └── app.log
```

### Local Development

1. **Edit code**:
```bash
cd ~/containers/yolo-detector
nano app/server.py
```

2. **Restart to apply changes**:
```bash
docker compose restart
```

3. **View logs**:
```bash
docker logs -f yolo-detector
```

4. **Test changes**:
```bash
curl -X POST http://localhost:9090/v1/vision/custom/yolov8x-oiv7 \
  -F "image=@test_images/person.jpg" | jq .
```

### Running Tests

**Single image test**:
```bash
curl -s -X POST http://localhost:9090/v1/vision/custom/yolov8x-oiv7 \
  -F "image=@test_images/person_dog.jpg" | jq '.count, .inference_ms'
```

**Multiple images test**:
```bash
for img in test_images/*.jpg; do
  echo "Testing: $img"
  curl -s -X POST http://localhost:9090/v1/vision/custom/yolov8x-oiv7 \
    -F "image=@$img" | jq '.count, .inference_ms'
done
```

**Concurrent requests test** (stress test):
```bash
for i in {1..10}; do
  curl -X POST http://localhost:9090/v1/vision/custom/yolov8x-oiv7 \
    -F "image=@test_images/person_dog.jpg" \
    -o /dev/null -s -w "Request $i: %{time_total}s\n" &
done
wait
```

### Monitoring

**Container stats**:
```bash
docker stats yolo-detector
```

**GPU monitoring**:
```bash
docker exec yolo-detector watch -n 1 nvidia-smi
```

**Live log monitoring**:
```bash
docker logs -f yolo-detector
```

**Filter logs for detections**:
```bash
docker logs -f yolo-detector | grep "Detection complete"
```

## Production Deployment

### Systemd Service

For automatic startup on system boot, install the systemd service:

```bash
# Copy service file
sudo cp yolo-detector.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable yolo-detector.service

# Start service
sudo systemctl start yolo-detector.service

# Check status
sudo systemctl status yolo-detector.service

# View logs
sudo journalctl -u yolo-detector.service -f
```

### Backup and Restore

**Backup configuration**:
```bash
tar -czf yolo-detector-backup.tar.gz \
  docker-compose.yml \
  .env \
  app/ \
  config/
```

**Restore**:
```bash
tar -xzf yolo-detector-backup.tar.gz
docker compose build
docker compose up -d
```

### Security Considerations

**Network Security**:
- BlueIrisSentry is designed for **local network use only**
- No built-in authentication (relies on network isolation)
- Recommend firewall rules to restrict access

**Restrict to Blue Iris machine only**:
```bash
sudo ufw allow from 192.168.1.8 to any port 9090 comment "Blue Iris PC"
sudo ufw deny 9090
```

**Docker security**:
- Container runs as non-root user
- GPU access is isolated through NVIDIA Container Runtime
- No privileged mode required

## Updating

### Update BlueIrisSentry

```bash
cd ~/containers/yolo-detector

# Pull latest changes
git pull

# Rebuild container
docker compose build

# Restart with new version
docker compose down
docker compose up -d

# Verify
curl http://localhost:9090/health | jq .version
```

### Update YOLOv8 Model

```bash
cd ~/containers/yolo-detector/models

# Backup current model
mv yolov8x-oiv7.pt yolov8x-oiv7.pt.backup

# Download new model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-oiv7.pt

# Restart container
docker compose restart
```

## License

This project is licensed under the **MIT License**.

### Third-Party Components

BlueIrisSentry uses the following open-source components:

- **Ultralytics YOLOv8**: [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **FastAPI**: [MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE)
- **PyTorch**: [BSD-3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- **CUDA Runtime**: NVIDIA proprietary (redistributable)
- **Open Images V7 Dataset**: [CC BY 4.0 License](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)

## Credits and Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- **Open Images V7**: [Google Research](https://storage.googleapis.com/openimages/web/index.html) - 600-class dataset
- **Blue Iris**: [Perspective Software](https://blueirissoftware.com/) - IP camera NVR software
- **NVIDIA**: CUDA runtime and GPU acceleration
- **FastAPI**: [Sebastián Ramírez](https://github.com/tiangolo/fastapi) - Modern Python web framework

## Support and Resources

### Documentation

- **BlueIrisSentry**: [CLAUDE.md](CLAUDE.md) - Complete implementation guide
- **YOLOv8**: https://docs.ultralytics.com/models/yolov8/
- **Blue Iris**: https://blueirissoftware.com/
- **Open Images V7**: https://storage.googleapis.com/openimages/web/index.html

### Community

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share your setup and experiences

### FAQ

**Q: Can I use this with other NVR software besides Blue Iris?**
A: Yes! Any software that supports CodeProject.AI format can use BlueIrisSentry. The API is compatible with standard object detection endpoints.

**Q: Do I need an NVIDIA RTX 6000 Pro?**
A: No. Any NVIDIA GPU with CUDA support will work. Performance will vary based on GPU capability. Tested on RTX 6000 Pro but should work on GTX 1060 and higher.

**Q: Can I run this without a GPU?**
A: Yes, but inference times will be significantly slower (500ms+ instead of 50-100ms). Not recommended for real-time multi-camera surveillance.

**Q: How many cameras can this handle simultaneously?**
A: Depends on your GPU and camera resolution. RTX 6000 Pro can easily handle 10+ simultaneous 1080p cameras with <100ms inference times.

**Q: Does this work on Windows?**
A: BlueIrisSentry is designed for Linux (Ubuntu). You could run it in WSL2 on Windows with NVIDIA GPU support, but native Linux is recommended.

**Q: Can I train custom models?**
A: Yes! Train your own YOLOv8 model using Ultralytics, place the `.pt` file in `models/`, and update `MODEL_NAME` in docker-compose.yml.

---

## Status

**Production Status**: ✅ **Fully Operational**
**Version**: 1.0.0
**Last Updated**: 2025-11-01
**Tested on**: NVIDIA RTX 6000 Pro Blackwell, Ubuntu 25.04, Blue Iris 5.6.9.8

**Successfully Tested**:
- ✅ Multi-camera surveillance (4K and 1080p)
- ✅ Real-time detection (50-115ms inference)
- ✅ Blue Iris integration (CodeProject.AI compatible)
- ✅ 600-class detection (Open Images V7)
- ✅ Auto-start on boot (systemd)
- ✅ Docker containerized deployment
- ✅ GPU acceleration (CUDA 13.0)

---

**Built with ❤️ for the Blue Iris community**

If BlueIrisSentry helped improve your surveillance system, consider giving it a ⭐ on GitHub!
