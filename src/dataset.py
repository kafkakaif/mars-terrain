import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

IGNORE = 255
SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_to_array(m):
    m = np.array(m)
    return m[..., 0] if m.ndim == 3 else m


class MarsData(Dataset):
    def __init__(self, split, indices, augment=False):
        self.ds = load_dataset("hassanjbara/AI4MARS", split=split)
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        row = self.ds[self.indices[k]]
        image = row["image"].convert("RGB")

        label = mask_to_array(row["label_mask"]).astype(np.int64)
        for key in ("rover_mask", "range_mask"):
            if row[key] is not None:
                label[mask_to_array(row[key]) > 0] = IGNORE

        # resize: bilinear for the image, nearest-neighbour for the mask
        image = image.resize((SIZE, SIZE), Image.BILINEAR)
        label = Image.fromarray(label.astype(np.uint8)).resize((SIZE, SIZE), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        label = np.array(label, dtype=np.int64)

        if self.augment:
            if np.random.rand() < 0.5:                      # flip image and mask together
                image = image[:, ::-1].copy()
                label = label[:, ::-1].copy()
            image = np.clip(image * np.random.uniform(0.8, 1.2), 0, 1)  # brightness, image only

        image = (image - MEAN) / STD
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, torch.from_numpy(label)


def get_labeled_indices(split):
    ds = load_dataset("hassanjbara/AI4MARS", split=split)
    return [i for i, h in enumerate(ds["has_labels"]) if h]


if __name__ == "__main__":
    idx = get_labeled_indices("train")
    print("Training images with labels:", len(idx))
    data = MarsData("train", idx[:10], augment=True)
    x, y = data[0]
    print("Image shape:", tuple(x.shape))
    print("Mask shape:", tuple(y.shape))
    print("Mask values:", torch.unique(y).tolist())