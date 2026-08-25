import threading
import time

_model = None
_camera = None
_camera_lock = threading.Lock()

def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
        print("[VISION] YOLOv8 loaded")
    return _model

def get_camera():
    global _camera
    import cv2
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            for idx in range(4):
                cam = cv2.VideoCapture(idx)
                if cam.isOpened():
                    _camera = cam
                    print(f"[VISION] Camera found at /dev/video{idx}")
                    return _camera
            print("[VISION] No camera found")
            return None
        return _camera

def detect_objects():
    """Capture a frame and return list of detected object names"""
    cam = get_camera()
    if cam is None:
        return []
    ret, frame = cam.read()
    if not ret:
        return []
    model = _load_model()
    results = model(frame, device=0, verbose=False)
    names = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                names.append(model.names[cls])
    return list(set(names))

def describe_scene():
    """Return a natural sentence describing what Ben sees"""
    objects = detect_objects()
    if not objects:
        return "I don't see anything clearly right now."
    if len(objects) == 1:
        return f"I can see a {objects[0]}."
    return "I can see " + ", ".join(objects[:-1]) + f", and a {objects[-1]}."
