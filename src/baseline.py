import numpy as np
from datasets import load_dataset

IGNORE = 255
CLASSES = ["soil", "bedrock", "sand", "big rock"]

def get_label(row):
    if row["label_mask"] is None:
        return None
    label = np.array(row["label_mask"])
    label = label[..., 0] if label.ndim == 3 else label
    label = label.astype(np.int64)
    for key in ("rover_mask", "range_mask"):
        if row[key] is not None:
            m = np.array(row[key])
            m = m[..., 0] if m.ndim == 3 else m
            label[m > 0] = IGNORE
    return label

def iou_per_class(preds, labels, n=4):
    cm = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(preds, labels):
        ok = t != IGNORE
        cm += np.bincount(n * t[ok] + p[ok], minlength=n * n).reshape(n, n)
    inter = np.diag(cm)
    return inter / (cm.sum(0) + cm.sum(1) - inter + 1e-9)

# 1. Class weights from the first 1000 training images
train = load_dataset("hassanjbara/AI4MARS", split="train")
counts = np.zeros(4)
for i in range(1000):
    label = get_label(train[i])
    if label is None:
        continue
    counts += np.bincount(label[label != IGNORE].ravel(), minlength=4)[:4]

share = counts / counts.sum()
weights = 1 - share
for name, s, w in zip(CLASSES, share, weights):
    print(f"{name:10s} share of pixels: {s:.3f}   weight: {w:.3f}")

# 2. Baseline: predict "soil" everywhere on the expert test set
test = load_dataset("hassanjbara/AI4MARS", split="test_min3")
labels = []
for row in test:
    label = get_label(row)
    if label is not None:
        labels.append(label)

preds = [np.zeros_like(l) for l in labels]
iou = iou_per_class(preds, labels)
print()
for name, v in zip(CLASSES, iou):
    print(f"Baseline IoU {name:10s}: {v:.3f}")
print(f"Baseline mean IoU: {iou.mean():.3f}")