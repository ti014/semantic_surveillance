"""
Utility helper functions
"""

import yaml
import os
import numpy as np
from typing import Dict, Tuple


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary chứa configuration
        
    Raises:
        FileNotFoundError: Nếu config file không tồn tại
        yaml.YAMLError: Nếu YAML format không hợp lệ
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: str) -> None:
    """
    Tạo directory nếu chưa tồn tại.
    
    Args:
        path: Đường dẫn directory cần tạo
    """
    os.makedirs(path, exist_ok=True)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 bounding boxes.
    
    Args:
        box1: Bounding box 1, format [x1, y1, x2, y2]
        box2: Bounding box 2, format [x1, y1, x2, y2]
        
    Returns:
        IoU score (0.0 - 1.0)
    """
    # Tọa độ của intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Diện tích intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Diện tích của mỗi box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Diện tích union
    union_area = box1_area + box2_area - intersection_area
    
    # Tránh chia cho 0
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou


def calculate_iou_with_region(box: np.ndarray, region: np.ndarray) -> float:
    """
    Tính IoU của box với một region cụ thể.
    
    Đây là wrapper của calculate_iou, nhưng semantic rõ ràng hơn
    khi sử dụng với person regions.
    
    Args:
        box: Equipment bounding box [x1, y1, x2, y2]
        region: Person region [x1, y1, x2, y2]
        
    Returns:
        IoU score (0.0 - 1.0)
    """
    return calculate_iou(box, region)


def box_center(box: np.ndarray) -> Tuple[float, float]:
    """
    Tính center point của bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple (center_x, center_y)
    """
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return center_x, center_y


def box_area(box: np.ndarray) -> float:
    """
    Tính diện tích của bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Diện tích (pixels)
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def is_point_in_box(point: Tuple[float, float], box: np.ndarray) -> bool:
    """
    Kiểm tra xem một điểm có nằm trong box hay không.
    
    Args:
        point: Tuple (x, y)
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        True nếu point nằm trong box
    """
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]
