from detect_brightness import detect_brightness

class_names = ['maybeStar', 'stars']

def get_yolo_to_json(results):
    boxes = results.boxes.xywh.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    return [
        {
            "class": class_names[c],
            "confidence": float(conf),
            "x": float(b[0]),
            "y": float(b[1]),
            "width": float(b[2]),
            "height": float(b[3]),
            "brightness": detect_brightness(results.orig_img, b[0], b[1], b[2], b[3])
        }
        for b, c, conf in zip(boxes, classes, confs)
    ]
