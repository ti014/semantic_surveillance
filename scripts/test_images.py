"""
Script to test PPE detection on a folder of images.
Useful for validation datasets or testing specific scenarios.
"""

import sys
import os
import argparse
import cv2
import glob
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import YOLOv8Detector, YOLOWorldDetector, PPEChecker, Visualizer
from utils import load_config, ensure_dir

def main():
    parser = argparse.ArgumentParser(description='Test PPE Pipeline on Images')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--source', type=str, required=True, help='Path to image folder')
    parser.add_argument('--output', type=str, default='output/images', help='Output folder')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    ensure_dir(args.output)

    # Initialize Modules
    print("[INFO] Initializing Pipeline...")
    
    # Detector Factory Logic
    detection_type = config.get('detection', {}).get('type', 'trained')
    if detection_type == 'trained':
        print("[INFO] Using YOLOv8Detector (Trained)")
        detector = YOLOv8Detector(config['detection']['model_path'], config)
    else:
        print("[INFO] Using YOLOWorldDetector (Zero-shot)")
        model_size = config['detection'].get('model_size', 's')
        
        # Check weights folder first, then root
        model_path = f'weights/yolov8{model_size}-world.pt'
        if not os.path.exists(model_path):
             model_path = f'yolov8{model_size}-world.pt' # Fallback
             
        detector = YOLOWorldDetector(model_path, config)
        
        # Set prompts for Zero-shot
        prompts = config['ppe_rules'].get('prompts', [])
        detector.set_prompts(prompts)

    checker = PPEChecker(config)
    visualizer = Visualizer(config)

    # Get Images
    image_paths = glob.glob(os.path.join(args.source, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"[INFO] Found {len(image_paths)} images in {args.source}")

    # Process
    for img_path in tqdm(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Resize if configured (optional, but consistent with video)
        if config['video'].get('resize_width'):
             h, w = frame.shape[:2]
             ratio = config['video']['resize_width'] / w
             new_h = int(h * ratio)
             frame = cv2.resize(frame, (config['video']['resize_width'], new_h))

        # 1. Detect
        iou_threshold = config['detection'].get('iou_threshold', 0.45)
        results = detector.predict(
            frame, 
            conf_threshold=config['detection']['confidence_threshold'], 
            iou_threshold=iou_threshold
        )
        detections = detector.get_detections(results)

        # 2. Check
        persons = detections.get('person', [])
        equipment = {k: v for k, v in detections.items() if k != 'person'}
        compliance_results = checker.check_compliance(persons, equipment)

        # 3. Visualize
        annotated_frame = visualizer.draw_results(frame, compliance_results, equipment)

        # Save
        filename = os.path.basename(img_path)
        save_path = os.path.join(args.output, filename)
        cv2.imwrite(save_path, annotated_frame)

    print(f"[INFO] Results saved to {args.output}")

if __name__ == '__main__':
    main()
