"""
ViolationLogger: Logging system cho violations
"""

import os
import json
from datetime import datetime
from typing import Dict, List
import cv2
import numpy as np
from utils.helpers import ensure_dir
from .ppe_checker import ComplianceResult


class ViolationLogger:
    """
    Logging violations với ảnh và metadata.
    
    Features:
    - Save violation frames as images
    - Log metadata in JSON Lines format
    - Track statistics
    - Optional console logging
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ViolationLogger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.violation_dir = config['logging']['violation_dir']
        self.save_images = config['logging']['save_images']
        self.console_log = config['logging']['console_log']
        
        # Ensure violation directory exists
        ensure_dir(self.violation_dir)
        
        # Log file path
        self.log_file = os.path.join(self.violation_dir, 'violations.jsonl')
        
        # Statistics
        self.stats = {
            'total_violations': 0,
            'violations_by_type': {}
        }
        
    def log_violation(
        self, 
        frame: np.ndarray, 
        result: ComplianceResult, 
        timestamp: datetime = None
    ) -> None:
        """
        Log một violation.
        
        Args:
            frame: Frame chứa violation
            result: ComplianceResult object
            timestamp: Timestamp của violation (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate image filename
        image_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = os.path.join(self.violation_dir, image_filename)
        
        # Save image nếu enabled
        if self.save_images:
            # Crop person region
            x1, y1, x2, y2 = map(int, result.person_box)
            
            # Add padding
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped_frame = frame[y1:y2, x1:x2]
            cv2.imwrite(image_path, cropped_frame)
        
        # Create metadata
        metadata = {
            'timestamp': timestamp.isoformat(),
            'bbox': result.person_box.tolist(),
            'missing_equipment': result.missing_equipment,
            'image_path': image_path if self.save_images else None
        }
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        # Update statistics
        self.stats['total_violations'] += 1
        for eq in result.missing_equipment:
            self.stats['violations_by_type'][eq] = \
                self.stats['violations_by_type'].get(eq, 0) + 1
        
        # Console log nếu enabled
        if self.console_log:
            print(f"[VIOLATION] {timestamp.strftime('%H:%M:%S')} - "
                  f"Missing: {', '.join(result.missing_equipment)}")
    
    def get_violation_stats(self) -> Dict:
        """
        Get violation statistics.
        
        Returns:
            Dictionary chứa statistics
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """
        Reset statistics counter.
        """
        self.stats = {
            'total_violations': 0,
            'violations_by_type': {}
        }
    
    def get_recent_violations(self, n: int = 10) -> List[Dict]:
        """
        Get n violations gần nhất.
        
        Args:
            n: Số lượng violations cần lấy
            
        Returns:
            List các violation metadata
        """
        if not os.path.exists(self.log_file):
            return []
        
        violations = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                violations.append(json.loads(line))
        
        # Return n violations cuối
        return violations[-n:]
