from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import io
import urllib.request, json
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch.nn as nn
from typing import List




app = FastAPI()





with open("labels.json") as f:
    labels = json.load(f)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load("model/celeb.pth", map_location="cpu"))
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

executor = ThreadPoolExecutor(max_workers=4)
jobs = {}  

metrics = {
    "total_requests": 0,
    "total_latency": 0.0,
    "queue_size" :0
}
metrics_lock = threading.Lock()

@app.post("/predict")
async def predict(file: UploadFile):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    async def run():
        with metrics_lock:
          metrics["total_requests"] += 1
          metrics["queue_size"] += 1
        start_time = time.time()

        jobs[job_id]["status"] = "processing"
        loop = asyncio.get_running_loop()
        
        with torch.no_grad():
            output = await loop.run_in_executor(executor, model, tensor)
        
        latency = time.time() - start_time

        with metrics_lock:
            metrics["total_latency"] += latency
            metrics["queue_size"] -= 1
            
        predicted_class = output.argmax(1).item()
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "class_id": predicted_class,
            "label": labels[predicted_class]
        }
        
    asyncio.create_task(run())
    return {"job_id": job_id}



@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    job_ids = []

    for file in files:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "result": None}

        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = preprocess(image).unsqueeze(0)

        async def run(job_id=job_id, tensor=tensor):
            with metrics_lock:
                metrics["total_requests"] += 1
                metrics["queue_size"] += 1

            jobs[job_id]["status"] = "processing"
            start = time.time()

            loop = asyncio.get_event_loop()
            with torch.no_grad():
                output = await loop.run_in_executor(executor, model, tensor)

            elapsed = time.time() - start

            with metrics_lock:
                metrics["total_latency"] += elapsed
                metrics["queue_size"] -= 1

            predicted_class = output.argmax(1).item()
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = {
                "class_id": predicted_class,
                "label": labels[predicted_class]
            }

        asyncio.create_task(run())
        job_ids.append(job_id)

    return {"job_ids": job_ids, "total": len(job_ids)}

@app.get("/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        return {"error": "job not found"}
    return jobs[job_id]

@app.get("/metrics")
def get_metrics():
    total = metrics["total_requests"]
    avg = metrics["total_latency"]/ total if total> 0 else 0
    return {
        "total_requests": total,
        "avg_latency_seconds": round(avg, 3),
        "queue_size": metrics["queue_size"],
        "worker_count": executor._max_workers
    }

@app.post("/jobs-status")
def jobs_status(job_ids: List[str]):
    return {
        job_id: jobs.get(job_id, {"error": "not found"})
        for job_id in job_ids
    }