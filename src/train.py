import random
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import MarsData, get_labeled_indices, IGNORE

random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = "cuda"
N_IMAGES, EPOCHS, BATCH = 3000, 10, 4

idx = get_labeled_indices("train")
random.shuffle(idx)
idx = idx[:N_IMAGES]
n_val = N_IMAGES // 10
val_idx, train_idx = idx[:n_val], idx[n_val:]

train_loader = DataLoader(MarsData("train", train_idx, augment=True),
                          batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(MarsData("train", val_idx), batch_size=BATCH, num_workers=0)

# class weights from your baseline step
weights = torch.tensor([0.647, 0.507, 0.857, 0.989], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)

model = smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", classes=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

def validate():
    model.eval()
    cm = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for p, t in zip(pred, y.numpy()):
                ok = t != IGNORE
                cm += np.bincount(4 * t[ok] + p[ok], minlength=16).reshape(4, 4)
    inter = np.diag(cm)
    return inter / (cm.sum(0) + cm.sum(1) - inter + 1e-9)

best = 0
with open("training_log.csv", "w") as log:
    log.write("epoch,loss,soil,bedrock,sand,big_rock,mean_iou\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        iou = validate()
        mean = iou.mean()
        print(f"epoch {epoch}  loss {total/len(train_loader):.3f}  val mean IoU {mean:.3f}  {np.round(iou, 3)}")
        log.write(f"{epoch},{total/len(train_loader):.4f},{','.join(f'{v:.4f}' for v in iou)},{mean:.4f}\n")
        log.flush()
        if mean > best:
            best = mean
            torch.save(model.state_dict(), "model.pt")
            print("  saved best model")