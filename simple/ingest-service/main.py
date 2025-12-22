import cv2

RTSP_URL = "rtsp://localhost:8554/camera1"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open RTSP stream")

print("✅ Connected to RTSP stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received")
        break

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

