import cv2
import mediapipe as mp

# ------------------ MediaPipe ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ------------------ Face Detection ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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

            # Draw nose point
            cv2.circle(frame, (nose_x, int(nose.y * h)), 5, (255, 0, 0), -1)

            if nose_x < w * 0.4:
                looking_away = True
                direction = "LOOKING LEFT"
            elif nose_x > w * 0.6:
                looking_away = True
                direction = "LOOKING RIGHT"

    # ------------------ ALERT SYSTEM ------------------
    if len(faces) == 0:
        alert = "NO FACE DETECTED"

    elif len(faces) > 1:
        alert = "MULTIPLE PEOPLE DETECTED"

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