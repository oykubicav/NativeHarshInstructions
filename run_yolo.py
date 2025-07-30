from ultralytics import YOLO

model = YOLO("best.pt")

def run_yolo(image_path: str):
    results = model(image_path)
    results[0].save(filename="result.jpg")
    return results[0]
