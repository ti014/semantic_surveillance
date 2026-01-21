from ultralytics import YOLO
import cv2

# Path to the trained model
model_path = r'runs/detect/runs/detect/ppe_detector/weights/best.pt'
video_path = 'bao-ho-lao-dong.mp4'

print(f"[INFO] Loading model from: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)

print(f"[INFO] Processing video: {video_path}")
print("[INFO] Press 'q' to stop.")

# Run inference and display
# source=video_path: input video
# show=True: display results window
# save=True: save output to runs/detect/predict/
# conf=0.25: confidence threshold
results = model.predict(source=video_path, show=True, save=True, conf=0.25)

print("[INFO] Done!")
