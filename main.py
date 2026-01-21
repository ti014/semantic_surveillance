"""
Dynamic PPE Compliance Monitoring System

Main application với video processing pipeline.
"""

import argparse
import time
from datetime import datetime
import cv2
import numpy as np

from core import YOLOv8Detector, YOLOWorldDetector, PPEChecker, ViolationLogger, Visualizer
from utils import load_config


class PPEMonitorApp:
    """
    Main application class.
    
    Features:
    - Video processing pipeline
    - Real-time monitoring
    - Keyboard controls
    - Statistics tracking
    """
    
    def __init__(self, config_path: str):
        """
        Initialize application.
        
        Args:
            config_path: Path to configuration file
        """
        print("[INFO] Loading configuration...")
        self.config = load_config(config_path)
        
        # Initialize modules
        print("[INFO] Initializing modules...")
        
        detection_type = self.config.get('detection', {}).get('type', 'trained')
        
        if detection_type == 'trained':
            # PRODUCTION PATH
            model_path = self.config['detection']['model_path']
            self.detector = YOLOv8Detector(model_path=model_path, config=self.config)
        
        else:
            # FUTURE/RESEARCH PATH (YOLO-World)
            model_size = self.config['detection'].get('model_size', 's')
            model_path = f'weights/yolov8{model_size}-world.pt'
            self.detector = YOLOWorldDetector(model_path=model_path, config=self.config)
            
            # Set prompts for Zero-shot
            prompts = self.config['ppe_rules'].get('prompts', [])
            self.detector.set_prompts(prompts)
        
        self.checker = PPEChecker(self.config)
        self.logger = ViolationLogger(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Video parameters
        self.source = self.config['video']['source']
        self.max_frames = self.config['video']['max_frames']
        self.resize_width = self.config['video']['resize_width']
        self.display_fps = self.config['video']['display_fps']
        
        # Statistics
        self.frame_count = 0
        self.total_persons = 0
        self.total_violations = 0
        
        # FPS tracking
        self.fps = 0.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # Control flags
        self.paused = False
        
    def run(self):
        """
        Main processing loop.
        """
        print(f"[INFO] Opening video source: {self.source}")
        cap = cv2.VideoCapture(self.source)
    def run(self, source: str = '0', max_frames: int = None, save_path: str = None):
        """
        Run main application loop.
        
        Args:
            source: Video source (camera index or file path)
            max_frames: Stop after N frames
            save_path: Path to save output video (e.g., output.mp4)
        """
        print(f"[INFO] Opening video source: {source}")
        if source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {source}")
            return
            
        print("[INFO] System started. Press 'q' to quit, 'p' to pause, 's' to save frame.")
        print("=" * 60)
        
        # Initialize VideoWriter if save_path is provided
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Determine output size
            if self.resize_width is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                target_width = self.resize_width
                target_height = int(height * (target_width / width))
            else:
                target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[INFO] Saving video to: {save_path} ({target_width}x{target_height} @ {fps:.1f} FPS)")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (target_width, target_height))
        
        try:
            while cap.isOpened():
                if not self.paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("[INFO] End of video stream.")
                        break
                    
                    # Convert RGBA to RGB if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Resize if needed
                    if self.resize_width is not None:
                        h, w = frame.shape[:2]
                        new_h = int(h * self.resize_width / w)
                        frame = cv2.resize(frame, (self.resize_width, new_h))
                    
                    # Process frame
                    annotated_frame = self._process_frame(frame)
                    
                    # Write to video
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Update FPS
                    self._update_fps()
                    
                    # Draw statistics
                    stats = {
                        'frame_number': self.frame_count,
                        'total_persons': self.total_persons,
                        'total_violations': self.total_violations
                    }
                    display_frame = self.visualizer.draw_statistics(
                        annotated_frame, 
                        stats, 
                        self.fps if self.display_fps else None
                    )
                    
                    # Display
                    cv2.imshow("PPE Compliance Monitor", display_frame)
                    
                    # Check max frames
                    if max_frames is not None and self.frame_count >= max_frames:
                        print(f"[INFO] Reached max frames: {max_frames}")
                        break
                else:
                    # Paused logic (optional: allows display update)
                    key = cv2.waitKey(100) & 0xFF 
                    if key == ord('p'):
                         self.paused = not self.paused
                    elif key == ord('q'):
                         break
                    continue
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("[INFO] Quit requested.")
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"[INFO] {status}")
                elif key == ord('s'):
                    self._save_frame(frame)
                    
        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"[INFO] Video output saved to {save_path}")
            cv2.destroyAllWindows()
            self._print_summary()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process một frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame
        """
        self.frame_count += 1
        
        # 1. Detection
        # Use config IoU for NMS (to reduce multiple boxes)
        iou_thres = self.config['detection'].get('iou_threshold', 0.45)
        
        results = self.detector.predict(
            frame, 
            conf_threshold=self.config['detection']['confidence_threshold'],
            iou_threshold=iou_thres
        )
        
        # 2. Parse detections
        detections = self.detector.get_detections(results)
        
        persons = detections.get('person', [])
        
        # Remove "person" từ equipment dict
        equipment = {k: v for k, v in detections.items() if k != 'person'}
        
        # 3. Check compliance
        compliance_results = self.checker.check_compliance(persons, equipment)
        
        # 4. Log violations
        current_violations = 0
        for result in compliance_results:
            if not result.is_compliant:
                self.logger.log_violation(frame, result, datetime.now())
                current_violations += 1
        
        # 5. Update statistics
        self.total_persons += len(persons)
        self.total_violations += current_violations
        
        # 6. Visualize
        annotated_frame = self.visualizer.draw_results(frame, compliance_results, equipment)
        
        # Console log
        if self.frame_count % 30 == 0:  # Log mỗi 30 frames
            print(f"Frame: {self.frame_count:04d} | "
                  f"FPS: {self.fps:.1f} | "
                  f"Persons: {len(persons)} | "
                  f"Violations: {current_violations}")
        
        return annotated_frame
    
    def _update_fps(self):
        """
        Update FPS calculation.
        """
        self.fps_frame_count += 1
        
        # Tính FPS mỗi 30 frames
        if self.fps_frame_count >= 30:
            elapsed_time = time.time() - self.fps_start_time
            self.fps = self.fps_frame_count / elapsed_time
            
            # Reset
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
    
    def _save_frame(self, frame: np.ndarray):
        """
        Save current frame.
        
        Args:
            frame: Frame to save
        """
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved frame: {filename}")
    
    def _print_summary(self):
        """
        Print summary statistics.
        """
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Frames:       {self.frame_count}")
        print(f"Total Persons:      {self.total_persons}")
        print(f"Total Violations:   {self.total_violations}")
        
        if self.total_persons > 0:
            violation_rate = (self.total_violations / self.total_persons) * 100
            print(f"Violation Rate:     {violation_rate:.1f}%")
        
        # Violation stats from logger
        violation_stats = self.logger.get_violation_stats()
        if violation_stats['violations_by_type']:
            print("\nViolations by Type:")
            for eq_type, count in violation_stats['violations_by_type'].items():
                print(f"  - {eq_type}: {count}")
        
        print("=" * 60)


def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(
        description='Dynamic PPE Compliance Monitoring System'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/ppe_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Video source (0 for webcam, or path to video file)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (for testing)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold override'
    )
    
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Path to save output video (e.g., output.mp4)'
    )
    
    args = parser.parse_args()
    
    # Initialize app
    app = PPEMonitorApp(args.config)
    
    # Override config logic (moving source logic to run method for consistency)
    source = args.source
    if source is None:
        source = app.config['video']['source']
        
    if args.conf is not None:
        app.config['detection']['confidence_threshold'] = args.conf
    
    # Run
    app.run(source=source, max_frames=args.max_frames, save_path=args.save_path)


if __name__ == '__main__':
    main()
