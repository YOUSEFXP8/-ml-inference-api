import aiohttp
import asyncio
import os
import time

val_dir = "val"
test_images = []

for celebrity in os.listdir(val_dir):
    celeb_path = os.path.join(val_dir, celebrity)
    if not os.path.isdir(celeb_path):
        continue
    imgs = os.listdir(celeb_path)
    if imgs:
        test_images.append(os.path.join(celeb_path, imgs[0]))
    if len(test_images) >= 20:
        break

async def single_user(session, image_path, user_id):
    with open(image_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=os.path.basename(image_path))
        async with session.post("http://127.0.0.1:8000/predict", data=data) as r:
            job_id = (await r.json())["job_id"]

    # poll until done
    start = time.time()
    while True:
        async with session.get(f"http://127.0.0.1:8000/job/{job_id}") as r:
            result = await r.json()
        if result["status"] == "done":
            elapsed = time.time() - start
            print(f"User {user_id} done in {elapsed:.2f}s — {result['result']['label']}")
            return elapsed
        await asyncio.sleep(0.1)

async def main():
    print(f"Simulating {len(test_images)} concurrent users\n")

    async with aiohttp.ClientSession() as session:
        start = time.time()
        times = await asyncio.gather(*[
            single_user(session, img, i+1)
            for i, img in enumerate(test_images)
        ])
        total = time.time() - start

    print(f"\nTotal wall time:  {total:.2f}s")
    print(f"Avg per user:     {sum(times)/len(times):.2f}s")
    print(f"Fastest user:     {min(times):.2f}s")
    print(f"Slowest user:     {max(times):.2f}s")

asyncio.run(main())