import cv2
import time

CASCADE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

cascade = cv2.CascadeClassifier(CASCADE)

if cascade.empty():
    print("ERROR: Cannot load Haar cascade")
    exit(1)

print("Camera opened")
print("Face detector loaded")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    h, w = frame.shape[:2]

    # Draw image centre
    cx = w // 2
    cy = h // 2

    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 2)

    # Draw detected faces
    for (x, y, fw, fh) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            2
        )

        face_cx = x + fw // 2
        face_cy = y + fh // 2

        # Face centre
        cv2.circle(
            frame,
            (face_cx, face_cy),
            5,
            (0, 0, 255),
            -1
        )

        # Line from image centre to face centre
        cv2.line(
            frame,
            (cx, cy),
            (face_cx, face_cy),
            (255, 0, 0),
            2
        )

        error_x = face_cx / w - 0.5
        error_y = face_cy / h - 0.5

        cv2.putText(
            frame,
            f"Face  error X:{error_x:+.2f} Y:{error_y:+.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Luna Face Tracking Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
