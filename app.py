"""
app.py — Gradio app for Google Cloud Run (or any Docker host).

Same detector as before, just adapted to listen on 0.0.0.0 and read the
port from the $PORT environment variable, which Cloud Run sets
automatically at runtime.
"""

import os
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("model/best.pt")   # must be included in the image
CONF_THRESHOLD = 0.25

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Copy your trained best.pt "
        "into the model/ folder before building the image."
    )

model = YOLO(str(MODEL_PATH))


def predict(image: Image.Image):
    if image is None:
        return None, "กรุณาอัปโหลดภาพก่อน"

    image = image.convert("RGB")
    results = model.predict(image, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]

    annotated_bgr = r.plot()          # numpy array, BGR
    annotated_rgb = annotated_bgr[..., ::-1]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append((model.names[cls_id], conf))
    detections.sort(key=lambda d: d[1], reverse=True)

    if detections:
        text = "\n".join(f"{label} — {conf * 100:.1f}%" for label, conf in detections)
    else:
        text = "Normal — ไม่พบรอยโรค (ไม่มีตำแหน่งที่โมเดลตรวจพบว่าผิดปกติ)"

    return annotated_rgb, text


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="อัปโหลดภาพอัลตราซาวด์ (.jpg / .png)"),
    outputs=[
        gr.Image(type="numpy", label="ผลการทำนาย"),
        gr.Textbox(label="รายละเอียด", lines=4),
    ],
    title="Breast Ultrasound Benign / Malignant Detector",
    description=(
        "อัปโหลดภาพอัลตราซาวด์เต้านม โมเดล YOLO จะตรวจหาตำแหน่งที่น่าจะเป็น "
        "benign หรือ malignant\n\n"
        "⚠️ เครื่องมือสาธิตเพื่อการศึกษาเท่านั้น ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์"
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
