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
    # Prefer the head tracker's most recent frame when it is running. The
    # tracker owns the camera continuously, and a second cam.read() from here
    # interleaves with its capture thread: each gets roughly half the frames,
    # cv2 reports nothing wrong, and this can be handed a frame taken mid-slew,
    # which is motion-blurred and describes the wall rather than the room.
    frame = None
    try:
        import head_tracker, time as _t
        t = head_tracker.tracker
        if getattr(t, "mode", "off") != "off":
            f, _pose, at = t._latest()
            # Only if genuinely fresh: a stale frame answers a question about
            # what was there seconds ago.
            if f is not None and (_t.time() - at) < 0.5:
                frame = f
    except Exception:
        frame = None

    if frame is None:
        # The tracker owns the camera. Ask it rather than opening the device:
        # two threads calling read() on one cv2.VideoCapture is not
        # thread-safe and on V4L2 blocks inside the driver holding the lock,
        # which hangs everything else that wants the camera.
        try:
            import head_tracker
            t = head_tracker.tracker
            if getattr(t, "_run", False):
                frame = t.grab_for_caller(timeout=2.5, max_age=0.5)
        except Exception:
            frame = None

    if frame is None:
        # Only reached when the tracker is not running at all, so there is no
        # second reader to collide with.
        cam = get_camera()
        if cam is None:
            return []
        # Warm up by wall clock. One read after the camera has been idle
        # returns a zeroed buffer — UVC drivers do this for the first few
        # dozen frames — and YOLO on a black image finds nothing, so this
        # reported "I don't see anything clearly" with a good view in front
        # of it.
        import time as _t2
        ret, frame = False, None
        deadline = _t2.time() + 2.0
        while _t2.time() < deadline:
            ok2, f2 = cam.read()
            if ok2 and f2 is not None:
                ret, frame = True, f2
                try:
                    if float(f2.mean()) >= 12.0:
                        break
                except Exception:
                    break
        if not ret or frame is None:
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
