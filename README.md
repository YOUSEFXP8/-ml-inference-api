---
title: ML Inference API
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---


# ML Inference API

A production-grade REST API for celebrity face recognition, built with FastAPI and ResNet50 transfer learning. Handles multiple concurrent users with parallel inference, async job tracking, real-time metrics, and full Docker support.

## Demo

Live API: https://Yousefxp8-ml-inference-api.hf.space/docs

## Features

- **Celebrity Recognition** — identifies 105 celebrities using ResNet50 fine-tuned with transfer learning
- **Parallel Inference** — ThreadPoolExecutor handles multiple concurrent requests simultaneously
- **Async Job Queue** — non-blocking job system with status tracking (pending → processing → done)
- **Batch Prediction** — submit multiple images in a single request
- **Real-time Metrics** — latency, request count, queue size, and worker count
- **Dockerized** — fully containerized for consistent deployment anywhere

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │           FastAPI Server         │
                        │                                  │
Client ──POST /predict──▶   Job Queue (in-memory dict)    │
       ◀── job_id ──────│         uuid per request         │
                        │               │                  │
Client ──GET /job/{id}──▶   ThreadPoolExecutor (4 workers) │
       ◀── status/result│               │                  │
                        │          ResNet50                │
Client ──GET /metrics───▶    (transfer learned, 105 classes)│
       ◀── latency/stats│                                  │
                        └─────────────────────────────────┘
```

**Request lifecycle:**
1. Client uploads image to `POST /predict`
2. Server creates a job ID and returns it instantly
3. Inference runs in background via ThreadPoolExecutor
4. Client polls `GET /job/{id}` until status is `done`
5. Result returned with predicted celebrity and class ID

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Submit single image, returns job ID immediately |
| GET | `/job/{id}` | Get job status and result |
| GET | `/metrics` | Real-time server performance metrics |
| POST | `/predict-batch` | Submit multiple images in one request |
| POST | `/jobs-status` | Check status of multiple jobs at once |

---

## Benchmark — 20 Concurrent Users

| Workers | Total Time | Avg Per User | Slowest User |
|---------|------------|--------------|--------------|
| 1 | 4.26s | 2.15s | 3.99s |
| 4 | 1.07s | 0.62s | 0.84s |

**4 workers delivers a 4x improvement in concurrent user throughput.**

With 1 worker, the slowest user waited 3.99s — queued behind 19 others. With 4 workers, the slowest user waited only 0.84s. The job queue ensures no request is dropped under load.

> Note: Sequential throughput shows minimal difference between worker counts due to GPU saturation — the RTX 4060 processes each inference in ~0.5s, faster than requests can queue. The bottleneck under concurrent load is CPU thread availability, not GPU throughput, which is where ThreadPoolExecutor provides the measurable gain.

---

## Model

| Property | Value |
|----------|-------|
| Architecture | ResNet50 |
| Backbone | Pretrained on ImageNet |
| Training method | Transfer learning — final layer replaced and fine-tuned |
| Dataset | 15,428 images across 105 celebrities |
| Validation accuracy | 78% |
| Output classes | 105 celebrities |

Transfer learning was chosen over training from scratch — freezing all ResNet layers except the final classification head reduces training time from days to minutes while achieving strong accuracy on a relatively small dataset.

---

## Parallelism Design

Model inference is CPU/GPU-bound. PyTorch releases the GIL during inference, meaning multiple threads genuinely run in parallel during heavy computation — unlike pure Python multithreading which is constrained by the GIL.

`ThreadPoolExecutor` with 4 workers was chosen after benchmarking:
- **asyncio alone** — insufficient for CPU-bound tasks, won't bypass the GIL
- **1 worker** — sequential, slowest user waits behind everyone else
- **4 workers** — 4x improvement in concurrent throughput, optimal for this workload
- **8+ workers** — diminishing returns, GPU becomes the bottleneck

This is the same pattern used in production ML serving systems — separate the I/O layer (FastAPI async) from the compute layer (thread pool).

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU recommended for training
- Dataset organized as one folder per celebrity

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-inference-api.git
cd ml-inference-api

# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Prepare dataset
# Add your images to train/ folder — one subfolder per celebrity:
# train/
#   Celebrity Name/
#     img1.jpg
#     img2.jpg

python split_dataset.py   # creates val/ split (80/20)
python train.py           # trains model, saves to model/celeb.pth

# Start server
python -m uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Docker

```bash
# Build
docker build -t ml-inference-api .

# Run
docker run -p 8000:8000 ml-inference-api
```

---

## Project Structure

```
ml-inference-api/
├── main.py              # FastAPI server — all endpoints
├── train.py             # Transfer learning training script
├── split_dataset.py     # Train/val dataset splitter
├── test_concurrent.py   # Concurrent user benchmark
├── test_batch.py        # Batch inference benchmark
├── labels.json          # Celebrity class names (105 classes)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI |
| ML framework | PyTorch |
| Model backbone | ResNet50 (torchvision) |
| Parallelism | ThreadPoolExecutor |
| Containerization | Docker |
| Image processing | Pillow |

---

## Interview Talking Points

**Why parallel processing?**
Reduces latency under concurrent load. Without it, users queue sequentially — the 20th user waits for all 19 before them. With 4 workers, queue depth drops 4x.

**What bottlenecks did you find?**
GIL limits pure Python parallelism, but PyTorch releases the GIL during inference enabling true multi-core execution. Under high concurrency the bottleneck shifts from CPU thread availability to GPU throughput.

**How would you scale this further?**
Replace in-memory job dict with Redis + Celery for a distributed task queue. Run multiple FastAPI instances behind a load balancer. Redis persists jobs across restarts and enables horizontal scaling across machines.

**ThreadPoolExecutor vs asyncio?**
asyncio handles I/O-bound concurrency — file reads, network calls. ThreadPoolExecutor handles CPU-bound tasks — model inference. This project uses both: asyncio for the request layer, ThreadPoolExecutor for the compute layer.

---

## Future Improvements

- [ ] Redis + Celery — replace in-memory queue with distributed task queue
- [ ] JWT authentication on `/predict`
- [ ] Top-3 predictions with confidence scores
- [ ] Fine-tune deeper ResNet layers for higher accuracy
- [ ] Horizontal scaling with load balancer