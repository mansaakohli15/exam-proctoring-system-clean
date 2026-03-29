import cv2
import mediapipe as mp
import numpy as np

# ------------------ MediaPipe ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ------------------ Face Detection ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ------------------ YOLO SETUP ------------------
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

with open("yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ------------------ Webcam ------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------ FACE DETECTION ------------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    # ------------------ HEAD DIRECTION ------------------
    results = face_mesh.process(rgb)

    looking_away = False
    direction = "LOOKING CENTER"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            # Draw nose point
            cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)

            if nose_x < w * 0.4:
                looking_away = True
                direction = "LOOKING LEFT"
            elif nose_x > w * 0.6:
                looking_away = True
                direction = "LOOKING RIGHT"

    # ------------------ YOLO OBJECT DETECTION ------------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    phone_detected = False

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes):
                label = classes[class_id]
            else:
                continue

            # 🔥 Lower confidence for better detection
            if confidence > 0.2:

                # Detect phone OR similar objects
                if label in ["cell phone", "remote", "book"]:
                    phone_detected = True

                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)

                    x = int(center_x - bw / 2)
                    y = int(center_y - bh / 2)

                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.putText(frame, "SUSPICIOUS OBJECT", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ------------------ ALERT SYSTEM ------------------
    if len(faces) == 0:
        alert = "NO FACE DETECTED"

    elif len(faces) > 1:
        alert = "MULTIPLE PEOPLE DETECTED"

    elif phone_detected:
        alert = "PHONE DETECTED"

    elif looking_away:
        alert = direction

    else:
        alert = "NORMAL"

    color = (0, 255, 0) if alert == "NORMAL" else (0, 0, 255)

    cv2.putText(frame, alert, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Exam Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()