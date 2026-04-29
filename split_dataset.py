import os
import shutil
import random

train_dir = 'train'
val_dir = 'val'
split = 0.2


for celebrity in os.listdir(train_dir):
    src = os.path.join(train_dir, celebrity)
    if not os.path.isdir(src):
        continue
    
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    val_count = max(1, int(len(images) * split))
    val_images = images[:val_count]

    val_celebrity = os.path.join(val_dir, celebrity)
    os.makedirs(val_celebrity, exist_ok=True)

    for img in val_images:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(val_celebrity, img)
        )

print("Done — val folder created")

    