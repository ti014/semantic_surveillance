"""
Visualizer: Render kết quả lên frame
"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from .ppe_checker import ComplianceResult


class Visualizer:
    """
    Visualization utilities cho PPE monitoring.
    
    Features:
    - Draw bounding boxes với màu theo compliance status
    - Add labels với missing equipment info
    - FPS counter
    - Statistics overlay
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Colors (BGR format)
        self.safe_color = tuple(config['visualization']['safe_color'])
        self.violation_color = tuple(config['visualization']['violation_color'])
        
        # Drawing parameters
        self.box_thickness = config['visualization']['box_thickness']
        self.font_scale = config['visualization']['font_scale']
        self.show_stats = config['visualization']['show_stats']
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_results(
        self, 
        frame: np.ndarray, 
        results: List[ComplianceResult],
        equipment: Dict[str, List[np.ndarray]] = None # NEW ARGUMENT
    ) -> np.ndarray:
        """
        Draw compliance results lên frame.
        """
        annotated_frame = frame.copy()
        
        # 1. Draw Equipment Boxes First (Behind Person Box)
        if equipment:
            for label, boxes in equipment.items():
                color = self.config['visualization']['colors'].get(label, (255, 255, 0))
                # OpenCV uses BGR
                if isinstance(color, list): color = tuple(color)
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    # Draw label slightly smaller
                    cv2.putText(annotated_frame, label, (x1, y1-5), self.font, 0.4, color, 1)

        # 2. Draw Person Boxes & Status
        for result in results:
            self._draw_single_result(annotated_frame, result)
        
        return annotated_frame
    
    def _draw_single_result(
        self, 
        frame: np.ndarray, 
        result: ComplianceResult
    ) -> None:
        """
        Draw một compliance result.
        
        Args:
            frame: Frame để draw (sẽ modify in-place)
            result: ComplianceResult object
        """
        # Unpack bounding box
        x1, y1, x2, y2 = map(int, result.person_box)
        
        # Chọn màu dựa trên compliance status
        color = self.safe_color if result.is_compliant else self.violation_color
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Tạo label
        if result.is_compliant:
            label = "SAFE"
        else:
            # Hiển thị thiết bị còn thiếu
            missing = ", ".join(result.missing_equipment)
            label = f"VIOLATION: Missing {missing}"
        
        # Tính kích thước text để vẽ background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, 
            self.font, 
            self.font_scale, 
            2
        )
        
        # Draw background cho text
        bg_y1 = max(0, y1 - text_height - baseline - 10)
        bg_y2 = y1
        cv2.rectangle(
            frame, 
            (x1, bg_y1), 
            (x1 + text_width + 10, bg_y2), 
            color, 
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),  # White text
            2
        )
    
    def draw_statistics(
        self, 
        frame: np.ndarray, 
        stats: Dict,
        fps: float = None
    ) -> np.ndarray:
        """
        Draw statistics overlay.
        
        Args:
            frame: Input frame
            stats: Statistics dictionary với keys:
                   - total_persons: int
                   - total_violations: int
                   - frame_number: int
            fps: Current FPS (optional)
            
        Returns:
            Frame với statistics overlay
        """
        if not self.show_stats:
            return frame
        
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Tạo text lines
        lines = []
        
        if 'frame_number' in stats:
            lines.append(f"Frame: {stats['frame_number']:04d}")
        
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        
        if 'total_persons' in stats:
            lines.append(f"Persons: {stats['total_persons']}")
        
        if 'total_violations' in stats:
            lines.append(f"Violations: {stats['total_violations']}")
        
        # Draw statistics panel
        panel_height = len(lines) * 30 + 20
        panel_width = 250
        
        # Semi-transparent background
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Draw text lines
        y_offset = 40
        for line in lines:
            cv2.putText(
                annotated_frame,
                line,
                (20, y_offset),
                self.font,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 30
        
        return annotated_frame
    
    def draw_fps(
        self, 
        frame: np.ndarray, 
        fps: float
    ) -> np.ndarray:
        """
        Draw FPS counter.
        
        Args:
            frame: Input frame
            fps: Current FPS
            
        Returns:
            Frame với FPS counter
        """
        annotated_frame = frame.copy()
        
        # Draw FPS
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated_frame,
            text,
            (10, 30),
            self.font,
            1.0,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
