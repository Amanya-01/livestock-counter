# app.py : Livestock Counter (cattle, goats, sheep) - runs on a normal PC or laptop
# Start with:  python app.py   (then the browser opens at http://127.0.0.1:7860)
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["Cattle", "Goat", "Sheep"]
BOX_BGR = [(44, 160, 44), (14, 127, 255), (180, 119, 31)]      # green, orange, blue

_STATE = {}


def load_config():
    cfg_path = HERE.parent / "config" / "config_d3.json"
    if not cfg_path.exists():
        raise FileNotFoundError("config_d3.json not found in the Proposed_model/config folder.")
    return json.loads(cfg_path.read_text())


def load_model():
    # the model is loaded once, on first use
    if "model" not in _STATE:
        from ultralytics import YOLO
        cfg = load_config()
        weights = HERE.parent / cfg["weights"]
        if not weights.exists():
            raise FileNotFoundError(f"Model file not found: {weights}")
        _STATE["model"] = YOLO(str(weights))
        _STATE["cfg"] = cfg
    return _STATE["model"], _STATE["cfg"]


def draw_boxes(img_bgr, pred):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    t = max(2, int(round(min(h, w) / 250)))
    for (x1, y1, x2, y2), c, cf in zip(pred["xyxyn"], pred["cls"], pred["conf"]):
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        col = BOX_BGR[int(c)]
        cv2.rectangle(out, p1, p2, col, t)
        cv2.putText(out, f"{CLASS_NAMES[int(c)]} {cf:.2f}", (p1[0] + 2, max(p1[1] - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(h, w) / 1000), col, max(1, t - 1), cv2.LINE_AA)
    return out


def analyse(image_rgb, use_tiling=True, conf=None, device=None):
    # image_rgb: numpy array (RGB) from the upload box. Returns (annotated RGB image, table rows, summary text)
    if image_rgb is None:
        return None, [], "Please upload a farm photo first."
    from pipeline import run_pipeline
    model, cfg = load_model()
    run_cfg = dict(imgsz=cfg["imgsz"], conf=float(conf if conf is not None else cfg["conf"]), iou=cfg.get("iou", 0.7),
                   tile=cfg.get("tile") if use_tiling else None, overlap=cfg.get("overlap", 0.25),
                   merge_iou=cfg.get("merge_iou") or 0.55, tile_max_area=cfg.get("tile_max_area"),
                   gate_k=cfg.get("gate_k"), gate_conf=cfg.get("gate_conf") or 0.3, device=device)
    img_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    pred, counts = run_pipeline(model, img_bgr, run_cfg)
    seconds = time.perf_counter() - t0
    annotated = cv2.cvtColor(draw_boxes(img_bgr, pred), cv2.COLOR_BGR2RGB)
    total = int(sum(counts.values()))
    rows = [[n, int(counts[n])] for n in CLASS_NAMES] + [["Total", total]]
    mode = "with tiling (slower, finds small animals)" if use_tiling else "whole photo only (faster)"
    summary = (f"Estimated animals: {total}  |  Cattle {counts['Cattle']}, Goats {counts['Goat']}, Sheep {counts['Sheep']}\n"
               f"Mode: {mode}  |  Confidence threshold: {run_cfg['conf']:.2f}  |  Time: {seconds:.1f} s\n"
               "These are estimates. In crowded herds some animals may be missed, so check the marked photo before recording the count.")
    return annotated, rows, summary


def build_ui():
    import gradio as gr
    cfg = load_config()
    with gr.Blocks(title="Livestock Counter") as demo:
        gr.Markdown("# Livestock Counter\nUpload a farm photo. The system marks cattle, goats and sheep and counts them.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Farm photo")
                tiling = gr.Checkbox(value=True, label="Use tiling (finds small or distant animals, but is slower)")
                conf = gr.Slider(0.10, 0.90, value=float(cfg["conf"]), step=0.05,
                                 label="Confidence threshold (higher = fewer false alarms, may miss animals)")
                btn = gr.Button("Count animals", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Result (green = cattle, orange = goat, blue = sheep)")
                table = gr.Dataframe(headers=["Animal", "Count"], datatype=["str", "number"], label="Counts")
                summary = gr.Textbox(label="Summary", lines=4)
        btn.click(analyse, inputs=[inp, tiling, conf], outputs=[out_img, table, summary])
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
