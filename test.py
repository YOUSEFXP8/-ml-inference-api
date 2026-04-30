import requests
import os
import time

# collect 200 images from across all celebrities
val_dir = "val"
all_images = []

for celebrity in os.listdir(val_dir):
    celeb_path = os.path.join(val_dir, celebrity)
    if not os.path.isdir(celeb_path):
        continue
    for img in os.listdir(celeb_path):
        all_images.append(os.path.join(celeb_path, img))
    if len(all_images) >= 200:
        break

print(f"Testing with {len(all_images)} images")

files = []
for path in all_images:
    files.append(("files", (os.path.basename(path), open(path, "rb"), "image/jpeg")))

start = time.time()
response = requests.post("http://127.0.0.1:8000/predict-batch", files=files)
data = response.json()
submit_time = time.time() - start
print(f"Submitted {data['total']} jobs in {submit_time:.2f}s")

time.sleep(10)  # give more time for 200 images

job_ids = data["job_ids"]
results = requests.post("http://127.0.0.1:8000/jobs-status", json=job_ids)
end = time.time()

done = sum(1 for r in results.json().values() if r["status"] == "done")
print(f"Completed: {done}/{len(job_ids)}")
print(f"Total time: {end - start:.2f}s")

metrics = requests.get("http://127.0.0.1:8000/metrics").json()
print(f"Avg latency: {metrics['avg_latency_seconds']}s")