"""
Test different YOLO-World model sizes
Compare accuracy: yolov8s vs yolov8m vs yolov8l
"""

import cv2
from core import OpenVocabDetector
import time

# Test config
prompts = ["person", "hard hat", "sleeveless reflective vest", "reflective safety vest"]
test_video = "bao-ho-lao-dong.mp4"
max_frames = 50

models = [
    ("yolov8s-world.pt", "Small (fastest)"),
    ("yolov8m-world.pt", "Medium (balanced)"),
    # ("yolov8l-world.pt", "Large (accurate)"),  # Uncomment if needed
]

results = {}

for model_path, model_name in models:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name} - {model_path}")
    print(f"{'='*70}")
    
    detector = OpenVocabDetector(model_path)
    detector.set_prompts(prompts)
    
    cap = cv2.VideoCapture(test_video)
    
    stats = {'person': 0, 'hard_hat': 0, 'safety_vest': 0}
    start_time = time.time()
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.predict(frame, conf_threshold=0.15, verbose=False)
        
        # Count
        if detections.boxes:
            for cls_id in detections.boxes.cls.cpu().numpy():
                cls_name = prompts[int(cls_id)].lower()
                
                if 'person' in cls_name:
                    stats['person'] += 1
                elif 'hat' in cls_name or 'helmet' in cls_name:
                    stats['hard_hat'] += 1
                elif 'vest' in cls_name:
                    stats['safety_vest'] += 1
    
    elapsed = time.time() - start_time
    fps = max_frames / elapsed
    
    cap.release()
    
    results[model_name] = {
        'stats': stats,
        'fps': fps,
        'time': elapsed
    }
    
    print(f"\nResults:")
    print(f"  Person: {stats['person']}")
    print(f"  Hard hat: {stats['hard_hat']}")
    print(f"  Safety vest: {stats['safety_vest']}")
    print(f"  Speed: {fps:.1f} FPS ({elapsed:.1f}s total)")

# Summary
print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print(f"{'='*70}")

for model_name, data in results.items():
    stats = data['stats']
    print(f"\n{model_name}:")
    print(f"  Accuracy: Hat={stats['hard_hat']}, Vest={stats['safety_vest']}")
    print(f"  Speed: {data['fps']:.1f} FPS")

# Recommendation
print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")

# Find best by hard hat + vest detections
best_model = max(results.items(), 
                 key=lambda x: x[1]['stats']['hard_hat'] + x[1]['stats']['safety_vest'])

print(f"Best accuracy: {best_model[0]}")
print(f"  Total PPE detected: {best_model[1]['stats']['hard_hat'] + best_model[1]['stats']['safety_vest']}")
print(f"  Speed: {best_model[1]['fps']:.1f} FPS")

if best_model[1]['fps'] < 5:
    print("\n⚠️ Warning: Slow for real-time. Consider smaller model.")
else:
    print("\n✅ Good balance of accuracy and speed!")
