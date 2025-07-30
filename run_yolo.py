from ultralytics import YOLO
import os
import tempfile

# Not: model dışarıda tanımlıysa reuse edebilirsin
model = YOLO("best.pt")

def run_yolo(image_path: str):
    results = model(image_path)

    # Güvenli geçici dosya yoluna kaydet
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp", delete=False) as f:
        results[0].save(filename=f.name)
    
    return results[0]
