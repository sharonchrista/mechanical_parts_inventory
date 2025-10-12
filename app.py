# app.py
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory
import torch

from utils.model import load_model_and_classes
from utils.infer import detect_objects
from utils.db import init_db, update_inventory, fetch_inventory

BASE_DIR   = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "fasterrcnn_best.pt"

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ---- Load model once at startup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, class_names = load_model_and_classes(MODEL_PATH, device=device)

# ---- DB init ----
init_db()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if not f:
        return "No file uploaded", 400

    save_path = UPLOAD_DIR / f.filename
    f.save(save_path.as_posix())

    vis_out = BASE_DIR / "static" / "predictions" / f"{save_path.stem}_pred.jpg"
    counts = detect_objects(model, save_path.as_posix(), class_names,
                            conf_th=0.5, iou_th=0.5, save_vis_path=vis_out)

    # Update DB with counts
    for name, qty in counts.items():
        if qty > 0:
            update_inventory(name, delta=qty)

    return render_template("result.html",
                           filename=f.filename,
                           counts=counts,
                           vis_rel_path=str(vis_out.relative_to(BASE_DIR)))

@app.route("/inventory", methods=["GET"])
def inventory():
    rows = fetch_inventory()
    return render_template("inventory.html", inventory=rows)

# serve uploaded files if ever needed
@app.route("/uploads/<path:fname>", methods=["GET"])
def get_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)

if __name__ == "__main__":
    app.run(debug=True)
