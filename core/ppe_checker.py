"""
PPE Compliance Logic Module.

Structure:
1. EquipmentAssociator: Handles spatial matching (geometry).
2. SafetyRuleValidator: Handles safety rules (business logic).
3. PPEChecker: Main Facade class.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# --- Data Structures ---
@dataclass
class ComplianceResult:
    """Result of compliance check for one person."""
    person_box: np.ndarray
    is_compliant: bool
    missing_equipment: List[str]
    detected_equipment: List[str]  # For debugging/visualization
    confidence: float = 1.0


# --- Component 1: Spatial Association Logic ---
class EquipmentAssociator:
    """
    Responsibility: Determine which equipment belongs to which person.
    Matches equipment boxes to person boxes based on spatial logic (IoU/IoE/Regions).
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.iou_threshold = config['detection'].get('iou_threshold', 0.45)
        
        # Region definitions (ratios relative to person height)
        regions_conf = config.get('regions', {})
        self.regions_ratios = {
            'head': regions_conf.get('head_ratio', 0.30),
            'torso': regions_conf.get('torso_ratio', 0.50),
            'legs': regions_conf.get('legs_ratio', 0.30)
        }
        
        # Mapping equipment -> body part
        self.equipment_region_map = config.get('regions', {}).get('equipment_map', {
            'hard_hat': 'head',
            'safety_vest': 'torso',
            'safety_boots': 'legs',
            'safety_goggles': 'head',
            'gloves': 'torso'
        })

    def associate_equipment(self, person_box: np.ndarray, all_equipment: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """
        Finds all equipment associated with a specific person.
        
        Args:
            person_box: [x1, y1, x2, y2]
            all_equipment: {'hard_hat': [[x1,y1,x2,y2], ...], ...}
            
        Returns:
            Dict of equipment belonging to THIS person: {'hard_hat': [box], ...}
        """
        person_equipment = {}
        person_regions = self._compute_regions(person_box)
        
        strict_regions_enabled = self.config.get('ppe_rules', {}).get('strict_regions', True)
        
        for eq_name, boxes in all_equipment.items():
            matched_boxes = []
            
            # Special Rule: Hard Hat ALWAYS strict check (Head Region)
            if eq_name == 'hard_hat':
                target_region = person_regions['head']
                use_strict = True
            elif strict_regions_enabled:
                region_key = self.equipment_region_map.get(eq_name, 'torso')
                target_region = person_regions.get(region_key, person_box)
                use_strict = True
            else:
                # Loose check (overlap with person body)
                target_region = person_box
                use_strict = False
                
            for box in boxes:
                if self._check_overlap(box, target_region, use_strict):
                    matched_boxes.append(box)
            
            if matched_boxes:
                person_equipment[eq_name] = matched_boxes
                
        return person_equipment

    def _compute_regions(self, person_box: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculates head, torso, legs sub-regions."""
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        
        # Head (Top N%)
        head_y2 = y1 + (h * self.regions_ratios['head'])
        
        # Torso (Middle N%) - Simplified approximation
        torso_y1 = head_y2
        torso_y2 = y2 - (h * 0.2) # Leaves 20% for legs at bottom
        
        # Legs (Bottom N%)
        legs_y1 = y2 - (h * self.regions_ratios['legs'])
        
        return {
            'head': np.array([x1, y1, x2, head_y2]),
            'torso': np.array([x1, torso_y1, x2, torso_y2]),
            'legs': np.array([x1, legs_y1, x2, y2])
        }

    def _check_overlap(self, box: np.ndarray, region: np.ndarray, strict_mode: bool) -> bool:
        """
        Checks if box overlaps with region.
        
        Strategies:
        1. Strict Mode: Requires significant overlap (IoU/Intersection).
        2. Loose Mode: Contains check or large IoE.
        """
        # Calculate Intersection
        xA = max(box[0], region[0])
        yA = max(box[1], region[1])
        xB = min(box[2], region[2])
        yB = min(box[3], region[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxArea = (box[2] - box[0]) * (box[3] - box[1])
        
        if boxArea == 0 or interArea == 0:
            return False
            
        # Metric: Intersection over Equipment Area (IoE)
        # How much of the equipment is inside the region?
        ioe = interArea / boxArea
        
        threshold = 0.3 if strict_mode else 0.1
        return ioe > threshold


# --- Component 2: Business Logic / Rules ---
class SafetyRuleValidator:
    """
    Responsibility: Check if associated equipment meets safety requirements.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.required_items = config['ppe_rules']['required_equipment'] # List[str] e.g. ['hard_hat', 'safety_vest']

    def validate(self, person_equipment: Dict[str, List[np.ndarray]]) -> Tuple[bool, List[str]]:
        """
        Validates safety based on what the person HAS.
        
        Returns:
            (is_compliant, missing_items)
        """
        missing = []
        for item in self.required_items:
            # Check if this required item exists in person's equipment
            if item not in person_equipment or not person_equipment[item]:
                missing.append(item)
        
        is_compliant = (len(missing) == 0)
        return is_compliant, missing


# --- Main Facade ---
class PPEChecker:
    """
    Main entry point. Orchestrates Association + Validation.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.associator = EquipmentAssociator(config)
        self.validator = SafetyRuleValidator(config)
        
        # Mapping logic (Normalization)
        self.class_mapping = self._build_mapping_table(config)

    def _build_mapping_table(self, config: Dict) -> Dict[str, str]:
        """Builds lookup table for class name normalization."""
        # 1. Config mapping
        mapping = config.get('ppe_rules', {}).get('class_mapping', {}).copy()
        
        # 2. Hardcoded logic (Backup)
        defaults = {
            'helmet': 'hard_hat',
            'vest': 'safety_vest',
            'boots': 'safety_boots',
            'person': 'person',
            'hard hat': 'hard_hat'
        }
        for k, v in defaults.items():
            if k not in mapping:
                mapping[k] = v
        return mapping

    def check_compliance(self, persons: List[np.ndarray], raw_equipment: Dict[str, List[np.ndarray]]) -> List[ComplianceResult]:
        """
        Process all persons and equipment.
        """
        # 1. Normalize Equipment Names
        normalized_eq = self._normalize_equipment(raw_equipment)
        
        results = []
        for person_box in persons:
            # 2. Association (Spatial Matching)
            # "Của ai đây?"
            person_items = self.associator.associate_equipment(person_box, normalized_eq)
            
            # 3. Validation (Rule Checking)
            # "Đủ đồ chưa?"
            is_valid, missing = self.validator.validate(person_items)
            
            # 4. Result
            results.append(ComplianceResult(
                person_box=person_box,
                is_compliant=is_valid,
                missing_equipment=missing,
                detected_equipment=list(person_items.keys())
            ))
            
        return results

    def _normalize_equipment(self, raw_equipment: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """Maps raw class names (e.g., 'helmet') to standard names (e.g., 'hard_hat')."""
        norm_eq = {}
        for name, boxes in raw_equipment.items():
            # Simple normalization
            clean_name = name.lower().strip()
            standard_name = self.class_mapping.get(clean_name, clean_name)
            
            if standard_name not in norm_eq:
                norm_eq[standard_name] = []
            norm_eq[standard_name].extend(boxes)
        return norm_eq
