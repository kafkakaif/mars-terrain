import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

train = load_dataset("hassanjbara/AI4MARS", split="train")
IGNORE = 255

def prepare(row):
    image = np.array(row["image"].convert("RGB"))
    label = np.array(row["label_mask"])
    label = label[..., 0] if label.ndim == 3 else label
    label = label.astype(np.int64)
    for key in ("rover_mask", "range_mask"):
        if row[key] is not None:
            m = np.array(row[key])
            m = m[..., 0] if m.ndim == 3 else m
            label[m > 0] = IGNORE
    return image, label

i = 0
while train[i]["label_mask"] is None:
    i += 1
print("Using image number", i)
img, label = prepare(train[i])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("Rover image")
ax[1].imshow(np.where(label == IGNORE, np.nan, label), cmap="tab10", vmin=0, vmax=9)
ax[1].set_title("Labels (blank = no label)")
plt.savefig("look.png")
plt.show()