# utils/infer.py
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    return t

@torch.inference_mode()
def detect_objects(model, image_path: str | Path, class_names: list[str],
                   conf_th: float = 0.5, iou_th: float = 0.5,
                   save_vis_path: str | Path | None = None) -> Dict[str, int]:
    """
    Returns dict like {"bearing": 2, "bolt": 1, ...}
    Optionally saves a visualization image if save_vis_path is given.
    """
    img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor_rgb(img)
    device = next(model.parameters()).device
    outputs = model([x.to(device)])[0]

    boxes  = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    # filter by confidence
    keep = scores >= conf_th
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # NMS
    if len(boxes) > 0:
        keep_idx = nms(boxes, scores, iou_th)
        boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]

    # count per class (labels are 1..N; 0 is background)
    counts: Dict[str, int] = {name: 0 for name in class_names}
    for lab in labels.tolist():
        if lab <= 0:
            continue
        name = class_names[lab - 1]
        counts[name] += 1

    # optional visualization
    if save_vis_path is not None:
        draw = ImageDraw.Draw(img)
        for b, lab, sc in zip(boxes.cpu().tolist(), labels.cpu().tolist(), scores.cpu().tolist()):
            if lab <= 0:
                continue
            name = class_names[lab - 1]
            x1, y1, x2, y2 = map(int, b)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), f"{name} {sc:.2f}", fill=(255, 0, 0))
        Path(save_vis_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_vis_path)

    return counts
