"""
Concept Drift Detector
======================
Monitors data distribution for concept drift events.
Wraps River's drift detection algorithms.
"""

import time
from typing import Optional
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger("models.drift_detector")


@dataclass
class DriftEvent:
    """A detected drift event."""
    timestamp: float
    detector_name: str
    severity: str  # "warning" or "critical"
    description: str


class DriftDetector:
    """
    Multi-method concept drift detection.
    
    Combines multiple detectors for robust drift identification:
    - ADWIN: Adaptive windowing (statistical)
    - DDM: Drift Detection Method (error rate monitoring)
    - EDDM: Early Drift Detection Method
    
    Response strategies:
    - "warning": Log and reduce confidence threshold
    - "critical": Alert, halt trading, retrain from buffer
    """
    
    def __init__(self, sensitivity: float = 0.001):
        self.sensitivity = sensitivity
        self.detectors = {}
        self.drift_events: list[DriftEvent] = []
        self.drift_active = False
        self.last_drift_time = 0.0
        
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize drift detection algorithms."""
        try:
            from river import drift
            self.detectors = {
                "adwin": drift.ADWIN(delta=self.sensitivity),
                "kswin": drift.KSWIN(alpha=self.sensitivity),
                "page_hinkley": drift.PageHinkley(threshold=self.sensitivity * 100),
            }
            logger.info(f"Drift detectors initialized: ADWIN, KSWIN, PageHinkley (sensitivity={self.sensitivity})")
        except ImportError:
            logger.warning("River not installed — drift detection disabled")
    
    def update(self, value: float) -> Optional[DriftEvent]:
        """
        Feed a new observation to all detectors.
        
        Args:
            value: The outcome to monitor (0 or 1 for binary)
        
        Returns:
            DriftEvent if drift detected, None otherwise.
        """
        if not self.detectors:
            return None
        
        max_severity = "none"
        detector_name = ""
        
        for name, detector in self.detectors.items():
            try:
                detector.update(value)
                
                if hasattr(detector, 'detected_change_flag'):
                    if detector.detected_change_flag:
                        if max_severity != "critical":
                            max_severity = "warning"
                            detector_name = name
                
                if hasattr(detector, 'drift_detected'):
                    if detector.drift_detected:
                        max_severity = "critical"
                        detector_name = name
                        
            except Exception as e:
                logger.debug(f"{name} detector error: {e}")
        
        if max_severity in ("warning", "critical"):
            self.drift_active = True
            self.last_drift_time = time.time()
            
            event = DriftEvent(
                timestamp=time.time(),
                detector_name=detector_name,
                severity=max_severity,
                description=f"Drift detected by {detector_name} ({max_severity})"
            )
            self.drift_events.append(event)
            
            # Keep only last 100 events
            if len(self.drift_events) > 100:
                self.drift_events = self.drift_events[-100:]
            
            if max_severity == "critical":
                logger.error(f"🚨 CRITICAL DRIFT: {event.description}")
            else:
                logger.warning(f"⚠️  Drift warning: {event.description}")
            
            return event
        
        return None
    
    def reset(self):
        """Reset all detectors (call after drift recovery)."""
        self.drift_active = False
        self._init_detectors()
        logger.info("Drift detectors reset")
    
    def summary(self) -> dict:
        return {
            "drift_active": self.drift_active,
            "total_events": len(self.drift_events),
            "last_drift_time": self.last_drift_time,
            "detectors_active": list(self.detectors.keys()),
        }
