# YOLOv8 Object Detection Server for Blue Iris - Complete Implementation Guide

## 🎯 Project Mission
Build a production-ready YOLOv8 object detection server optimized for Blue Iris IP camera integration, running on Ubuntu Server with NVIDIA RTX 6000 Pro Blackwell GPU (96GB GDDR7).

---

## 📋 Table of Contents
1. [System Environment](#system-environment)
2. [Critical Blue Iris Integration Architecture](#critical-blue-iris-integration-architecture)
3. [YOLOv8 Model Configuration](#yolov8-model-configuration)
4. [Project Structure](#project-structure)
5. [API Specifications](#api-specifications)
6. [Docker Configuration](#docker-configuration)
7. [Implementation Phases](#implementation-phases)
8. [Blue Iris Setup Guide](#blue-iris-setup-guide)
9. [Testing & Verification](#testing--verification)
10. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Environment

### Hardware Specifications
- **GPU:** NVIDIA RTX PRO 6000 Blackwell
  - 96GB GDDR7 memory
  - Compute Capability 9.0
  - CUDA 13.0
- **CPU:** AMD EPYC 64-core
- **RAM:** 384GB ECC DDR4
- **OS:** Ubuntu 25.04 (Plucky Puffin)
- **Driver:** 580.95.05 (open-source kernel module - **REQUIRED** for Blackwell)

### Software Stack
- **Docker:** Compose V2
- **Base Image:** `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` (proven on this system)
- **Python:** 3.11+ (via Docker)
- **PyTorch:** Latest with CUDA 12.x support (compatible with CUDA 13.0 via compat layer)
- **Ports:** 
  - API: `9080` (all standard ports already in use)
  - Web UI: `9081`

### ⚠️ Critical Blackwell/CUDA 13.0 Compatibility Notes
**MANDATORY REQUIREMENTS:**
- Must use open-source NVIDIA drivers (proprietary causes dependency conflicts)
- Base image must be `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` (tested working)
- Set `ENV PIP_BREAK_SYSTEM_PACKAGES=1` for Ubuntu 24.04+ Python packaging
- PyTorch CUDA 12.x wheels are forward-compatible with CUDA 13.0
- Many pre-built images don't support Compute Capability 9.0 - build custom
- Test GPU access with `nvidia-smi` inside container BEFORE proceeding

---

## 🔌 CRITICAL: Blue Iris Integration Architecture

### Understanding the Motion Detection Workflow

**Blue Iris sends FULL FRAME images to your AI server, NOT cropped motion regions!**

#### Complete Workflow:
```
1. Motion Detection (Blue Iris)
   ↓
   Blue Iris motion sensor triggers (bug flies by camera)
   Blue Iris zones, object size, travel distance applied
   
2. Snapshot Capture (Blue Iris)
   ↓
   Blue Iris captures FULL FRAME image (entire camera view)
   Image contains: moving bug + static car + background + everything visible
   
3. Image Transmission (Blue Iris → Your Server)
   ↓
   POST request to http://192.168.1.3:9080/detect
   Content-Type: multipart/form-data
   Field name: "image"
   NO crop, NO motion region coordinates, NO metadata
   
4. AI Analysis (Your Server)
   ↓
   YOLOv8 processes ENTIRE image
   Detects ALL objects: ["insect", "car", "person", "tree", ...]
   Returns bounding boxes for EVERYTHING detected
   
5. Response (Your Server → Blue Iris)
   ↓
   JSON response with ALL detections:
   {
     "success": true,
     "predictions": [
       {"label": "insect", "confidence": 0.65, "x_min": 100, ...},
       {"label": "car", "confidence": 0.95, "x_min": 500, ...},
       {"label": "person", "confidence": 0.88, "x_min": 800, ...}
     ]
   }
   
6. Validation & Filtering (Blue Iris)
   ↓
   Blue Iris applies its own logic:
   ✓ Is detected object in configured object whitelist? (person/car only)
   ✓ Did object move minimum distance? (bug moved 5px, car 0px)
   ✓ Is object in correct zone? (Zone A = driveway only)
   ✓ Is object above size threshold? (bug too small)
   ✓ Has object changed from previous frame? (car static)
   
7. Decision (Blue Iris)
   ↓
   CONFIRM: Person detected in driveway, moved 100px
   CANCEL: Bug not in whitelist, car didn't move
```

### 🐛 Your Carport Bug Scenario - Explained

**Scenario:** Bug flies in front of carport camera where your car is parked.

**What happens:**
1. Bug triggers Blue Iris motion sensor
2. Blue Iris captures full frame (includes bug + car + carport)
3. Blue Iris sends entire image to your AI server
4. Your server detects: `insect: 65%`, `car: 95%`
5. Your server returns BOTH detections
6. Blue Iris receives both detections
7. **Blue Iris cancels alert** because:
   - `insect` not in object whitelist (only person/car/truck configured)
   - `car` hasn't moved from previous frame (static object filtering)
   - Bug didn't travel minimum distance (5 pixels vs. 100 required)
   - Bug below minimum object size threshold

### Your Server's Responsibilities
✅ **DO THIS:**
- Accept full frame images via POST
- Detect ALL objects in the image
- Return accurate bounding boxes for everything
- Be fast (<100ms per image)
- Provide high confidence scores

❌ **DON'T DO THIS:**
- Try to determine motion regions
- Filter detections based on zones
- Track object movement between frames
- Compare current frame to previous
- Make decisions about what matters
- Second-guess your detections

**Blue Iris handles ALL the intelligence about what objects matter!**

### Why This Design Makes Sense
- **Separation of concerns:** Detection server = detect, Blue Iris = decide
- **Flexibility:** Users configure filtering in Blue Iris, not server code
- **Simplicity:** Your server is intentionally "dumb" - just detect accurately
- **Reusability:** Same server works for any Blue Iris user with different needs
- **Performance:** No complex logic in detection pipeline

---

## 🤖 YOLOv8 Model Configuration

### Model Selection: YOLOv8x with Open Images V7

**Model:** `yolov8x-oiv7.pt` (YOLOv8 Extra Large, Open Images V7 pretrained)

**Why This Model:**
- **Best Accuracy:** Highest mAP of all YOLOv8 variants
- **600 Classes:** Comprehensive object coverage for security scenarios
- **Your Hardware:** 96GB VRAM easily handles the largest model
- **Target Performance:** <100ms inference achievable on RTX 6000 Pro
- **Proven Dataset:** Trained on 16M professionally annotated bounding boxes

### Model Statistics
- **Dataset:** Open Images V7
- **Classes:** 600 object categories
- **Training Images:** 1.74M images with bounding boxes
- **Parameters:** ~68M (YOLOv8x)
- **Model Size:** ~136MB download
- **Input Size:** 640x640 (standard)

### 600 Classes Coverage - Security-Relevant Subset

The model detects 600 classes. Here are the most relevant for security/surveillance:

#### People (17 classes)
Person, Man, Woman, Boy, Girl, Human body, Human face, Human arm, Human leg, Human hand, Human foot, Human head, Human hair, Human nose, Human mouth, Human ear, Human eye

#### Vehicles (28 classes)
Car, Van, Truck, Bus, Taxi, Limousine, Ambulance, Vehicle, Motorcycle, Bicycle, Skateboard, Golf cart, Wheelchair, Snowmobile, Go-kart, Airplane, Helicopter, Boat, Canoe, Jet ski, Submarine, Tank, Train, Tram

#### Animals & Wildlife (40+ classes)
Dog, Cat, Horse, Cattle, Pig, Sheep, Goat, Chicken, Turkey, Rabbit, Deer, Fox, Bear, Squirrel, Raccoon, Skunk, Mouse, Rat, Bird, Eagle, Owl, Duck, Goose, Swan, Lizard, Snake, Turtle, Tortoise, Frog, Fish, Butterfly, Insect, Spider, Bee, Ant

#### Security-Relevant Objects (15 classes)
Backpack, Suitcase, Handbag, Briefcase, Luggage, Package, Box, Weapon (note: limited), Knife, Sword, Firearm (note: limited), Tool, Ladder, Barrel, Container

**NOTE:** All 600 classes will be available. Users can filter in Blue Iris configuration.

### Class Filtering Implementation
- **Server Side:** Load ALL 600 classes, detect everything
- **Configuration File:** `/app/config/classes_filter.json` (optional)
- **Blue Iris Side:** Users configure object whitelist in Blue Iris settings
- **Default Behavior:** Return all detections, let Blue Iris filter

---

## 📁 Project Structure

```
~/containers/yolo-detector/
├── docker-compose.yml          # Main orchestration file
├── Dockerfile                  # Custom build with CUDA 13.0 support
├── .env                        # Environment variables
├── yolo-detector.service       # Systemd service for auto-start
├── install.sh                  # Automated setup script (optional)
├── README.md                   # Complete documentation
├── CLAUDE.md                   # This file (for Claude Code reference)
│
├── app/                        # Application code (mounted as volume)
│   ├── server.py               # FastAPI application (main entry point)
│   ├── detector.py             # YOLOv8 detection logic
│   ├── config.py               # Configuration management
│   ├── utils.py                # Helper functions
│   ├── web_ui.html             # Testing interface
│   └── requirements.txt        # Python dependencies
│
├── models/                     # Model weights (auto-downloaded)
│   └── yolov8x-oiv7.pt         # Downloaded on first run (~136MB)
│
├── config/                     # Configuration files
│   ├── classes_filter.json     # Optional class filtering
│   └── server_config.json      # Server settings
│
├── test_images/                # Many more sample test images in this folder
│   ├── person.jpg
│   ├── car.jpg
│   └── person_dog.jpg
│
├── output/                     # Detection results (debug images)
│   └── detections/
│
└── logs/                       # Application logs
    ├── app.log
    ├── inference.log
    └── error.log
```

---

## 🔌 API Specifications

### Base URL
```
http://192.168.1.3:9080
```

### 1. `/detect` - Primary Detection Endpoint

**Method:** POST  
**Content-Type:** multipart/form-data  
**Authentication:** None (local network)

**Request:**
```bash
curl -X POST http://192.168.1.3:9080/detect \
  -F "image=@test.jpg" \
  -F "confidence=0.25"
```

**Request Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | File | Yes | - | JPEG/PNG image file |
| confidence | Float | No | 0.25 | Minimum confidence threshold (0.0-1.0) |

**Response (Success):**
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
  "inference_ms": 45,
  "process_ms": 52,
  "model": "yolov8x-oiv7",
  "image_size": [1920, 1080],
  "timestamp": "2025-11-01T12:34:56.789Z"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Invalid image format",
  "message": "Image must be JPEG or PNG",
  "code": "INVALID_FORMAT"
}
```

### 2. `/health` - Health Check Endpoint

**Method:** GET

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 6000 Pro Blackwell",
  "gpu_memory_total_mb": 98304,
  "gpu_memory_used_mb": 2048,
  "model_loaded": true,
  "model_name": "yolov8x-oiv7.pt",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "cuda_version": "13.0"
}
```

### 3. `/models/info` - Model Information

**Method:** GET

**Response:**
```json
{
  "model_name": "yolov8x-oiv7",
  "model_path": "/app/models/yolov8x-oiv7.pt",
  "model_size_mb": 136.2,
  "num_classes": 600,
  "input_size": 640,
  "device": "cuda:0",
  "half_precision": false,
  "batch_size": 1,
  "warmup_complete": true,
  "total_detections": 12847,
  "avg_inference_ms": 47.3
}
```

### 4. `/classes` - List Available Classes

**Method:** GET

**Response:**
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
    "people": ["Person", "Man", "Woman", "Boy", "Girl"],
    "vehicles": ["Car", "Truck", "Van", "Bus", "Motorcycle", "Bicycle"],
    "animals": ["Dog", "Cat", "Horse", "Bird", "Deer"]
  }
}
```

### 5. `/` - Web UI (GET)

Serves the testing web interface at `http://192.168.1.3:9081/`

---

## 🐳 Docker Configuration

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements
COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/output /app/config

# Expose ports
EXPOSE 9080 9081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:9080/health || exit 1

# Run server
CMD ["python3", "server.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  yolo-detector:
    build: .
    container_name: yolo-detector
    hostname: yolo-detector
    restart: unless-stopped
    
    ports:
      - "9080:9080"  # API endpoint
      - "9081:9081"  # Web UI
    
    volumes:
      - ./app:/app
      - ./models:/app/models
      - ./config:/app/config
      - ./test_images:/app/test_images
      - ./output:/app/output
      - ./logs:/app/logs
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - MODEL_NAME=yolov8x-oiv7.pt
      - CONFIDENCE_THRESHOLD=0.25
      - API_PORT=9080
      - WEB_UI_PORT=9081
      - LOG_LEVEL=INFO
      - MAX_QUEUE_SIZE=10
      - SAVE_DEBUG_IMAGES=false
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### requirements.txt

```txt
# Core dependencies
ultralytics>=8.0.200
torch>=2.0.0
torchvision>=0.15.0

# API framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Image processing
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
```

### Systemd Service File

**File:** `/etc/systemd/system/yolo-detector.service`

```ini
[Unit]
Description=YOLOv8 Object Detection Server for Blue Iris
Requires=docker.service
After=docker.service nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/raru/containers/yolo-detector

# Wait for NVIDIA devices
ExecStartPre=/bin/bash -c 'until [ -e /dev/nvidia-uvm ]; do sleep 1; done'
ExecStartPre=/bin/bash -c 'until [ -e /dev/nvidia0 ]; do sleep 1; done'

# Start container
ExecStart=/usr/bin/docker compose up -d

# Stop container
ExecStop=/usr/bin/docker compose down

TimeoutStartSec=120
User=raru
Group=raru

[Install]
WantedBy=multi-user.target
```

---

## 🔨 Implementation Phases

### Phase 1: Project Setup & Docker Build (10 min)

**Actions:**
1. Create complete directory structure
2. Create `Dockerfile` with CUDA 13.0 compatibility
3. Create `requirements.txt` with all dependencies
4. Create `.env` file with configuration
5. Build Docker image
6. Verify GPU access inside container

**Verification:**
```bash
# Build image
docker compose build

# Start container
docker compose up -d

# Check GPU access
docker exec yolo-detector nvidia-smi

# Expected output: GPU visible with correct driver version
```

**Success Criteria:**
- ✅ Container builds without errors
- ✅ `nvidia-smi` shows RTX 6000 Pro Blackwell
- ✅ CUDA 13.0 visible
- ✅ No driver version mismatches

### Phase 2: Core Detection Logic (15 min)

**Files to Create:**
- `app/detector.py` - YOLOv8 model wrapper
- `app/config.py` - Configuration management
- `app/utils.py` - Helper functions

**detector.py Implementation:**
```python
from ultralytics import YOLO
import torch
import time

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self._warmup()
    
    def _warmup(self):
        # Run dummy inference to initialize
        dummy_img = torch.zeros(1, 3, 640, 640).to(self.device)
        self.model(dummy_img, verbose=False)
    
    def detect(self, image_path, conf=None):
        conf = conf or self.conf_threshold
        start_time = time.time()
        
        results = self.model(image_path, conf=conf, verbose=False)
        
        inference_time = (time.time() - start_time) * 1000
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls]
                
                detections.append({
                    "label": label,
                    "confidence": round(conf_score, 2),
                    "x_min": int(x1),
                    "y_min": int(y1),
                    "x_max": int(x2),
                    "y_max": int(y2)
                })
        
        return {
            "detections": detections,
            "inference_ms": round(inference_time, 1)
        }
```

**Verification:**
```bash
# Test detection with sample image
docker exec yolo-detector python3 -c "
from detector import YOLODetector
detector = YOLODetector('/app/models/yolov8x-oiv7.pt')
result = detector.detect('/app/test_images/person.jpg')
print(result)
"
```

**Success Criteria:**
- ✅ Model downloads yolov8x-oiv7.pt automatically
- ✅ GPU inference confirmed (check logs for CUDA device)
- ✅ Detection returns valid JSON
- ✅ Inference time <100ms

### Phase 3: FastAPI Server (20 min)

**Files to Create:**
- `app/server.py` - Main API application

**server.py Key Features:**
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from detector import YOLODetector
import asyncio
from collections import deque
import time

app = FastAPI(title="YOLOv8 Detection Server")

# CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None
request_queue = deque(maxlen=10)

@app.on_event("startup")
async def startup_event():
    global detector
    detector = YOLODetector("/app/models/yolov8x-oiv7.pt")

@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    # Save uploaded image
    image_path = f"/tmp/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())
    
    # Detect
    result = detector.detect(image_path, conf=confidence)
    
    return {
        "success": True,
        "predictions": result["detections"],
        "count": len(result["detections"]),
        "inference_ms": result["inference_ms"],
        "model": "yolov8x-oiv7"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": detector is not None
    }
```

**Verification:**
```bash
# Test health endpoint
curl http://localhost:9080/health | jq .

# Test detection endpoint
curl -X POST http://localhost:9080/detect \
  -F "image=@test_images/person.jpg" \
  | jq .
```

**Success Criteria:**
- ✅ `/health` returns 200 OK
- ✅ `/detect` accepts images and returns detections
- ✅ Response format matches Blue Iris requirements
- ✅ Multiple concurrent requests handled

### Phase 4: Web UI (10 min)

**File:** `app/web_ui.html`

**Key Features:**
- Image upload (drag-and-drop)
- Confidence threshold slider
- Canvas with bounding box overlay
- Detection results table
- Inference time display

**Verification:**
```bash
# Open in browser
firefox http://192.168.1.3:9081/
```

**Success Criteria:**
- ✅ Upload image successfully
- ✅ Detections displayed with bounding boxes
- ✅ Labels and confidence scores shown
- ✅ Inference time displayed

### Phase 5: Systemd Service (5 min)

**Actions:**
1. Copy service file to `/etc/systemd/system/`
2. Reload systemd daemon
3. Enable service for auto-start
4. Start service
5. Verify status

**Commands:**
```bash
sudo cp yolo-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-detector.service
sudo systemctl start yolo-detector.service
sudo systemctl status yolo-detector.service
```

**Verification:**
```bash
# Check service status
sudo systemctl status yolo-detector.service

# Reboot and verify auto-start
sudo reboot
# After reboot
curl http://localhost:9080/health
```

**Success Criteria:**
- ✅ Service starts without errors
- ✅ Service auto-starts on boot
- ✅ API accessible after reboot

### Phase 6: Testing & Documentation (10 min)

**Actions:**
1. Test with multiple concurrent requests
2. Measure GPU utilization
3. Create comprehensive README
4. Document Blue Iris integration steps
5. Add troubleshooting section

**Performance Testing:**
```bash
# Concurrent request test
for i in {1..10}; do
  curl -X POST http://localhost:9080/detect \
    -F "image=@test_images/person.jpg" &
done
wait

# Monitor GPU usage
watch -n 1 nvidia-smi
```

**Success Criteria:**
- ✅ Handles 10 concurrent requests
- ✅ GPU utilization visible
- ✅ No memory leaks
- ✅ Documentation complete

---

## 🎥 Blue Iris Setup Guide

### Prerequisites
- Blue Iris v5.6.9.8 or later installed on Windows PC
- YOLOv8 detection server running at `http://192.168.1.3:9080`
- Cameras added to Blue Iris

### Step 1: Global AI Configuration

1. Open Blue Iris
2. Click Settings (gear icon)
3. Go to **AI** tab
4. **Uncheck** "Use AI server on localhost"
5. Enter custom AI server URL:
   - **URL:** `http://192.168.1.3:9080`
   - **API Path:** `/detect`
6. Click **Test** to verify connection
7. Select **Start** to enable AI processing

### Step 2: Per-Camera Configuration

For each camera you want AI detection on:

1. Right-click camera → **Camera Properties**
2. Go to **Trigger** tab

#### Motion Sensor Configuration:
1. Check **Motion Sensor**
2. Click **Configure**
3. Set **Minimum object size:** 500-1000 pixels (adjust per camera)
4. Set **Minimum contrast:** 10-15
5. Set **Make time:** 0.1-0.2 seconds
6. Configure **Zones** to exclude areas you don't want monitored:
   - Zone A = areas to monitor (highlight in green)
   - Non-highlighted areas = ignored for motion
7. Click **OK**

#### Artificial Intelligence Configuration:
1. In **When Triggered** section, check **Artificial Intelligence**
2. Click **Artificial Intelligence** to configure
3. **AI configuration settings:**
   - Check **Enabled**
   - **Objects to detect:** `person,car,truck,van,motorcycle,bicycle,dog,cat`
   - **Min confidence:** 40% (adjust based on results)
   - **Real-time images:** 10-15 (more = slower but more accurate)
   - **Analyze one each:** 500ms
   - **Begin analysis with:** Motion leading images
   - Uncheck **Use main stream if available** (slower, not more accurate)
4. Click **OK**

#### Alert Configuration:
1. Go to **Alerts** tab
2. Check **Add to the alerts list**
3. Check **JPEG files** (for AI analysis)
4. Configure actions (email, push notification, etc.)
5. In **To cancel** field: `nothing found:0` (cancels if no objects detected)

### Step 3: Testing

1. Walk in front of camera
2. Check **Blue Iris Status → AI** tab
3. Verify detections appear with:
   - Object label (person/car/etc.)
   - Confidence percentage
   - Inference time (<100ms ideal)
   - Green checkmark = confirmed

### Step 4: Optimization Tips

**If detection is too slow:**
- Reduce **Real-time images** to 5-10
- Increase **Analyze one each** to 1000ms
- Make sure **Use main stream** is unchecked

**If getting false positives:**
- Increase **Min confidence** to 50-60%
- Adjust motion sensor **minimum object size**
- Use zones to exclude problem areas
- Refine object list to only what you need

**If missing detections:**
- Lower **Min confidence** to 30-35%
- Increase **Real-time images** to 20
- Check motion sensor sensitivity
- Verify object is in the detection list

### Step 5: Advanced Configuration

**Static Object Filtering:**
- Blue Iris → Camera Settings → AI tab
- **Uncheck** "Detect/Ignore static objects" (creates constant API calls)
- Let your server detect everything, Blue Iris will ignore static objects

**Zone-Based Object Crossing:**
1. Configure zones (A, B, C, etc.)
2. Go to Motion Sensor → Object Detection
3. Set **Object crosses zones:** A→B (trigger only when object moves from A to B)

**Custom Models (Future):**
- Your server uses Open Images V7 (600 classes)
- Blue Iris can filter to specific object lists
- Future: Add custom trained models for specific use cases

---

## ✅ Testing & Verification

### Manual Testing Checklist

**Container Tests:**
```bash
# 1. Build and start
cd ~/containers/yolo-detector
docker compose build
docker compose up -d

# 2. Check GPU access
docker exec yolo-detector nvidia-smi
# Expected: RTX 6000 Pro visible, CUDA 13.0

# 3. Check logs
docker logs -f yolo-detector
# Expected: "Model loaded", "Server started on port 9080"

# 4. Health check
curl http://localhost:9080/health | jq .
# Expected: "status": "healthy", "gpu_available": true

# 5. Model info
curl http://localhost:9080/models/info | jq .
# Expected: "num_classes": 600, "device": "cuda:0"

# 6. Test detection
curl -X POST http://localhost:9080/detect \
  -F "image=@test_images/person.jpg" \
  | jq .
# Expected: detections returned, inference_ms < 100

# 7. Web UI
firefox http://192.168.1.3:9081/
# Expected: Upload page loads, detection works
```

**Performance Tests:**
```bash
# Concurrent requests (10 simultaneous)
for i in {1..10}; do
  curl -X POST http://localhost:9080/detect \
    -F "image=@test_images/person.jpg" \
    -o /dev/null -s -w "Request $i: %{time_total}s\n" &
done
wait

# GPU utilization monitoring
watch -n 1 nvidia-smi

# Container resource usage
docker stats yolo-detector
```

**Blue Iris Integration Test:**
1. Configure camera in Blue Iris (see setup guide)
2. Walk in front of camera
3. Check Blue Iris Status → AI tab
4. Verify detection appears with correct label
5. Check inference time (<100ms)
6. Verify alert triggers correctly

### Automated Testing Script

**File:** `test.sh`

```bash
#!/bin/bash

echo "Testing YOLOv8 Detection Server..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test health endpoint
echo -n "Testing /health endpoint... "
HEALTH=$(curl -s http://localhost:9080/health | jq -r '.status')
if [ "$HEALTH" == "healthy" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test GPU availability
echo -n "Testing GPU availability... "
GPU=$(curl -s http://localhost:9080/health | jq -r '.gpu_available')
if [ "$GPU" == "true" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test detection endpoint
echo -n "Testing /detect endpoint... "
DETECT=$(curl -s -X POST http://localhost:9080/detect \
  -F "image=@test_images/person.jpg" \
  | jq -r '.success')
if [ "$DETECT" == "true" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test inference speed
echo -n "Testing inference speed... "
INFERENCE_TIME=$(curl -s -X POST http://localhost:9080/detect \
  -F "image=@test_images/person.jpg" \
  | jq -r '.inference_ms')
if (( $(echo "$INFERENCE_TIME < 100" | bc -l) )); then
    echo -e "${GREEN}✓ PASS ($INFERENCE_TIME ms)${NC}"
else
    echo -e "${RED}✗ FAIL ($INFERENCE_TIME ms > 100ms)${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All tests passed!${NC}"
```

---

## 🔧 Troubleshooting

### Issue: Container Won't Start

**Symptoms:**
- `docker compose up -d` fails
- Container exits immediately

**Diagnosis:**
```bash
docker compose logs yolo-detector
docker ps -a | grep yolo-detector
```

**Solutions:**
1. **GPU device not found:**
   ```bash
   # Check NVIDIA devices exist
   ls -l /dev/nvidia*
   
   # Restart nvidia-persistenced
   sudo systemctl restart nvidia-persistenced
   
   # Wait for devices
   sleep 5
   ```

2. **CUDA version mismatch:**
   - Verify base image: `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`
   - Check CUDA in container: `docker exec yolo-detector nvcc --version`

3. **Port already in use:**
   ```bash
   # Check what's using port 9080
   sudo lsof -i :9080
   
   # Change port in docker-compose.yml if needed
   ```

### Issue: Model Download Fails

**Symptoms:**
- "Failed to download model" in logs
- Container starts but `/detect` fails

**Solutions:**
1. **Internet connectivity:**
   ```bash
   docker exec yolo-detector ping -c 3 github.com
   ```

2. **Manual download:**
   ```bash
   cd ~/containers/yolo-detector/models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   # Rename and restart
   docker compose restart
   ```

3. **Disk space:**
   ```bash
   df -h ~/containers/yolo-detector/models
   ```

### Issue: Slow Inference Times (>100ms)

**Symptoms:**
- Detection works but inference_ms > 100

**Diagnosis:**
```bash
# Check GPU usage
docker exec yolo-detector nvidia-smi

# Check if using CPU instead of GPU
docker logs yolo-detector | grep "device"
```

**Solutions:**
1. **Verify GPU acceleration:**
   - Look for "Using device: cuda:0" in logs
   - If "Using device: cpu", GPU not being used

2. **Model not on GPU:**
   - Check detector.py ensures `model.to(device)`
   - Restart container

3. **System overload:**
   ```bash
   # Check CPU usage
   docker stats yolo-detector
   
   # Check if other processes using GPU
   nvidia-smi
   ```

### Issue: Blue Iris Can't Connect

**Symptoms:**
- "Connection refused" in Blue Iris
- Timeouts in Blue Iris AI tab

**Diagnosis:**
```bash
# Test from host machine
curl http://192.168.1.3:9080/health

# Test from Blue Iris machine
curl http://192.168.1.3:9080/health
```

**Solutions:**
1. **Firewall blocking:**
   ```bash
   # Ubuntu firewall
   sudo ufw status
   sudo ufw allow 9080/tcp
   ```

2. **Container not listening on all interfaces:**
   - Check server.py: `uvicorn.run(host="0.0.0.0")`
   - Restart container

3. **Wrong IP address:**
   - Verify server IP: `ip addr show`
   - Update Blue Iris URL

### Issue: Memory Leaks / Container Crashes

**Symptoms:**
- Container restarts periodically
- OOM (Out of Memory) errors

**Diagnosis:**
```bash
# Check memory usage
docker stats yolo-detector

# Check logs for OOM
docker logs yolo-detector | grep -i "memory\|killed"
```

**Solutions:**
1. **Too many queued requests:**
   - Reduce `MAX_QUEUE_SIZE` in environment variables
   - Add rate limiting in FastAPI

2. **Debug images not cleaned:**
   - Check `output/` directory size
   - Disable `SAVE_DEBUG_IMAGES` if enabled

3. **Memory leak in code:**
   - Check for unclosed file handles
   - Review detector.py for proper cleanup

### Issue: Wrong Detections / Low Accuracy

**Symptoms:**
- Objects not detected (false negatives)
- Wrong objects detected (false positives)

**Solutions:**
1. **Lower confidence threshold:**
   - Blue Iris: Set "Min confidence" to 30-35%
   - API: Use `confidence=0.3` parameter

2. **Image quality issues:**
   - Check camera resolution (should be 640x640+)
   - Verify lighting conditions
   - Test with high-quality sample images

3. **Object not in training set:**
   - Check 600 classes list
   - Some objects may not be well-represented

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, restart container |
| `RuntimeError: CUDA error: no kernel image` | CUDA version mismatch | Use correct base image (12.9.1) |
| `ModuleNotFoundError: No module named 'ultralytics'` | Dependencies not installed | Rebuild container |
| `OSError: [Errno 28] No space left on device` | Disk full | Clean up old images/logs |
| `Connection refused` | Server not listening | Check if container running |
| `JSONDecodeError` | Invalid API response | Check API format matches Blue Iris |

---

## 📚 Additional Resources

### Documentation Links
- **YOLOv8 Ultralytics:** https://docs.ultralytics.com/models/yolov8/
- **Open Images V7:** https://storage.googleapis.com/openimages/web/index.html
- **FastAPI:** https://fastapi.tiangolo.com/
- **Blue Iris:** https://blueirissoftware.com/

### Performance Benchmarks (Expected)
- **Inference Time:** 40-80ms per image (RTX 6000 Pro)
- **Throughput:** 12-25 FPS (single image)
- **GPU Memory:** ~4GB VRAM used
- **Concurrent Requests:** 10+ simultaneous without degradation

### Future Enhancements
- [ ] Batch processing for multiple images
- [ ] Custom model training integration
- [ ] Prometheus metrics endpoint
- [ ] Video stream support
- [ ] Object tracking between frames
- [ ] License plate recognition module
- [ ] Face recognition module
- [ ] MQTT alert publishing
- [ ] Home Assistant integration
- [ ] Dashboard with statistics

---

## 🎯 Success Criteria Summary

**Phase 1: Infrastructure**
- ✅ Container builds with CUDA 13.0 support
- ✅ GPU visible inside container (`nvidia-smi` works)
- ✅ All directories created correctly

**Phase 2: Core Functionality**
- ✅ Model downloads automatically
- ✅ Detection works on sample images
- ✅ GPU inference confirmed (logs show cuda:0)
- ✅ Inference time <100ms

**Phase 3: API Integration**
- ✅ `/detect` endpoint accepts images
- ✅ Response format matches Blue Iris requirements
- ✅ `/health` endpoint returns GPU status
- ✅ Concurrent requests handled

**Phase 4: Blue Iris Integration**
- ✅ Blue Iris connects successfully
- ✅ Detections appear in Blue Iris alerts
- ✅ Confidence scores displayed
- ✅ Alerts trigger correctly

**Phase 5: Production Readiness**
- ✅ Systemd service auto-starts on boot
- ✅ Logs rotating properly
- ✅ Error handling robust
- ✅ Documentation complete

---

## 🚀 Quick Start Commands

```bash
# 1. Navigate to project directory
cd ~/containers/yolo-detector

# 2. Build and start container
docker compose build
docker compose up -d

# 3. Check GPU access
docker exec yolo-detector nvidia-smi

# 4. View logs
docker logs -f yolo-detector

# 5. Test health endpoint
curl http://localhost:9080/health | jq .

# 6. Test detection
curl -X POST http://localhost:9080/detect \
  -F "image=@test_images/person.jpg" \
  | jq .

# 7. Open web UI
firefox http://192.168.1.3:9081/

# 8. Install systemd service
sudo cp yolo-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-detector.service
sudo systemctl start yolo-detector.service

# 9. Check service status
sudo systemctl status yolo-detector.service

# 10. Monitor performance
watch -n 1 'docker stats yolo-detector --no-stream'
```

---

## 📝 Notes for Claude Code

**Priority Tasks (Execute in Order):**
1. Create complete directory structure
2. Implement `detector.py` with GPU support
3. Implement `server.py` with FastAPI
4. Create `web_ui.html` for testing
5. Test detection with sample images
6. Create systemd service
7. Write comprehensive README.md
8. Test Blue Iris integration

**Critical Reminders:**
- Test GPU access FIRST before proceeding
- Verify model downloads correctly
- Ensure response format matches Blue Iris exactly
- Test concurrent requests before declaring complete
- Create README with Blue Iris configuration steps

**Code Quality Standards:**
- Use type hints in all Python functions
- Add comprehensive docstrings
- Include error handling in all API endpoints
- Log all errors with tracebacks
- Follow PEP 8 style guide

**Testing Requirements:**
- Unit tests not required (production focus)
- Manual testing checklist must pass
- Performance benchmarks must be documented
- Blue Iris integration must be verified

---

**End of CLAUDE.md** 🎉

Ready to build a production-ready YOLOv8 detection server!