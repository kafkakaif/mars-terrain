import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp
from PIL import Image

SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLORS = np.array([[130, 130, 130],   # soil
                   [150, 90, 50],     # bedrock
                   [255, 220, 0],     # sand
                   [220, 30, 30]])    # big rock

model = smp.DeepLabV3Plus("resnet34", encoder_weights=None, classes=4)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def predict(image):
    img = Image.fromarray(image).convert("RGB").resize((SIZE, SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        pred = model(x).argmax(1)[0].numpy()
    overlay = 0.55 * np.array(img) + 0.45 * COLORS[pred]
    return overlay.astype(np.uint8)

gr.Interface(predict, gr.Image(), gr.Image(),
             title="Mars Terrain Segmentation",
             description="Upload a rover image. Gray = soil, brown = bedrock, yellow = sand, red = big rock.").launch()