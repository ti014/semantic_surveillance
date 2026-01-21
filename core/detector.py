"""
Detection Module.

Architecture:
- BaseDetector (ABC): Abstract Interface.
- YOLOv8Detector: Production implementation for Fine-tuned models.
- YOLOWorldDetector: Research implementation for Zero-shot/Open Vocabulary.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

class BaseDetector(ABC):
    """Abstract Base Class for PPE Detectors."""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.model_path = model_path
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
    
    @abstractmethod
    def predict(self, frame: np.ndarray, conf_threshold: float, iou_threshold: float) -> Results:
        """Perform inference."""
        pass
    
    @abstractmethod
    def get_detections(self, results: Results) -> Dict[str, List[np.ndarray]]:
        """Parse results into standardized format."""
        pass


class YOLOv8Detector(BaseDetector):
    """
    Implementation for Fine-tuned YOLOv8 (Production).
    Uses standard class IDs from training dataset (COCO-like or Custom).
    """
    def __init__(self, model_path: str, config: Dict):
        super().__init__(model_path, config)
        self.class_mapping = config['ppe_rules'].get('class_mapping', {})
        print("[INFO] Detector Mode: YOLOv8 (Trained/Production)")

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> Results:
        # Standard YOLO inference
        results = self.model.predict(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False
        )
        return results[0]

    def get_detections(self, results: Results) -> Dict[str, List[np.ndarray]]:
        """
        Parses detections and maps class names using config mapping.
        """
        detections = {}
        
        # Initialize lists for known mapped classes
        for mapped_name in self.class_mapping.values():
            detections[mapped_name] = []
            
        boxes = results.boxes
        if boxes is None:
            return detections
            
        for box in boxes:
            cls_id = int(box.cls[0])
            internal_name = results.names[cls_id] # e.g., 'helmet'
            
            # Map 'helmet' -> 'hard_hat'
            final_name = self.class_mapping.get(internal_name, internal_name)
            
            xyxy = box.xyxy[0].cpu().numpy()
            
            if final_name not in detections:
                detections[final_name] = []
            detections[final_name].append(xyxy)
            
        return detections


class YOLOWorldDetector(BaseDetector):
    """
    Implementation for YOLO-World (Zero-shot/Future).
    Uses dynamic prompts for open vocabulary detection.
    """
    def __init__(self, model_path: str, config: Dict):
        super().__init__(model_path, config)
        print("[INFO] Detector Mode: YOLO-World (Zero-shot/Research)")
        self.prompts = []
        
    def set_prompts(self, prompts: List[str]):
        """Set custom text prompts for the model."""
        self.prompts = prompts
        self.model.set_classes(prompts)
        print(f"[INFO] Set prompts: {prompts}")

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.1, iou_threshold: float = 0.45) -> Results:
        # YOLO-World inference
        results = self.model.predict(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False
        )
        return results[0]

    def get_detections(self, results: Results) -> Dict[str, List[np.ndarray]]:
        """
        Parses detections based on set prompts.
        """
        detections = {prompt: [] for prompt in self.prompts}
        
        boxes = results.boxes
        if boxes is None:
            return detections
            
        for box in boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(self.prompts):
                cls_name = self.prompts[cls_id]
                xyxy = box.xyxy[0].cpu().numpy()
                detections[cls_name].append(xyxy)
                
        return detections
