import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import MarsData, get_labeled_indices, MEAN, STD, IGNORE

CLASSES = ["soil", "bedrock", "sand", "big rock"]
device = "cuda"
model = smp.DeepLabV3Plus("resnet34", encoder_weights=None, classes=4).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

def evaluate(split):
    idx = get_labeled_indices(split)
    data = MarsData(split, idx)
    cm = np.zeros((4, 4), dtype=np.int64)
    sand_miss, k = [], 0
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=4):
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for p, t in zip(pred, y.numpy()):
                ok = t != IGNORE
                cm += np.bincount(4 * t[ok] + p[ok], minlength=16).reshape(4, 4)
                sand = t == 2
                if sand.sum() > 200:
                    sand_miss.append(((p[sand] != 2).mean(), k))
                k += 1
    inter = np.diag(cm)
    iou = inter / (cm.sum(0) + cm.sum(1) - inter + 1e-9)
    acc = inter.sum() / cm.sum()
    return iou, acc, data, sand_miss

rows = []
for split in ["test_min3", "test_min1"]:
    iou, acc, data, sand_miss = evaluate(split)
    print(f"\n{split}")
    for n, v in zip(CLASSES, iou):
        print(f"  IoU {n:9s}: {v:.3f}")
    print(f"  mean IoU: {iou.mean():.3f}   pixel accuracy: {acc:.3f}")
    rows.append((split, iou, acc))
    if split == "test_min3" and sand_miss:
        worst = sorted(sand_miss, reverse=True)[:6]
        fig, ax = plt.subplots(len(worst), 3, figsize=(10, 3 * len(worst)))
        for r, (miss, k) in enumerate(worst):
            x, y = data[k]
            with torch.no_grad():
                p = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
            img = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
            t = np.where(y.numpy() == IGNORE, np.nan, y.numpy())
            ax[r, 0].imshow(img); ax[r, 0].set_title(f"sand missed: {miss:.0%}")
            ax[r, 1].imshow(t, cmap="tab10", vmin=0, vmax=9); ax[r, 1].set_title("true labels")
            ax[r, 2].imshow(p, cmap="tab10", vmin=0, vmax=9); ax[r, 2].set_title("prediction")
            for a in ax[r]: a.axis("off")
        plt.tight_layout()
        plt.savefig("sand_errors.png")

with open("results.csv", "w") as f:
    f.write("split,soil,bedrock,sand,big_rock,mean_iou,pixel_acc\n")
    for split, iou, acc in rows:
        f.write(f"{split},{','.join(f'{v:.3f}' for v in iou)},{iou.mean():.3f},{acc:.3f}\n")