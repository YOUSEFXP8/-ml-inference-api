from fastapi import FastAPI, UploadFile
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



app = FastAPI()

model =models.resnet50(pretrained=True)
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

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as f:
    labels = json.load(f)

executor = ThreadPoolExecutor(max_workers=4)
jobs = {}  

metrics = {
    "total_requests": 0,
    "total_latency": 0.0,
    "queue_size" :0
}

@app.post("/predict")
async def predict(file: UploadFile):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    async def run():
        metrics["total_requests"] += 1
        metrics["queue_size"] += 1
        start_time = time.time()

        jobs[job_id]["status"] = "processing"
        loop = asyncio.get_event_loop()
        
        with torch.no_grad():
            output = await loop.run_in_executor(executor, model, tensor)
        
        latency = time.time() - start_time
        metrics_lock = threading.Lock()

# then inside run():
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