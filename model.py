# utils/model.py
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

def load_model_and_classes(model_path: str | Path, device: str = "cpu"):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    ckpt = torch.load(p, map_location=device)
    class_names = ckpt.get("class_names", None)
    num_classes = ckpt.get("num_classes", None)

    if class_names is None or num_classes is None:
        raise RuntimeError("Checkpoint missing 'class_names' or 'num_classes'.")

    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, class_names
