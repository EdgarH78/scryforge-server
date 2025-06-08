import cv2
import logging
import time
import numpy as np
from typing import Iterator, Optional
from .lens import Lens, ProcessedFrame

logger = logging.getLogger(__name__)


class Camera:
    _instances = {}  # Class variable to store camera instances

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None        
        self.max_retries = 3
        self.lenses = []

    @classmethod
    def get_camera(cls, device_id: int):    
        """Get or create a camera instance"""
        if device_id not in cls._instances:
            camera = cls(device_id)
            if camera.initialize():
                cls._instances[device_id] = camera
        return cls._instances.get(device_id)

    @classmethod
    def get_available_cameras(cls, max_devices: int = 10) -> list[int]:
        """Get list of available camera device IDs"""
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
            else:
                break
        return available

    def is_available(self):
        """Check if camera is available"""
        return self.cap is not None and self.cap.isOpened()

    @classmethod
    def cleanup_all(cls):
        """Release all cameras"""
        for camera in cls._instances.values():
            camera.release()
        cls._instances.clear()

    def initialize(self) -> bool:
        """Initialize camera connection"""
        if self.cap is not None:
            self.release()
        self.cap = cv2.VideoCapture(self.device_id)
        return self.cap.isOpened()
    
    def add_lens(self, lens: Lens):
        self.lenses.append(lens)

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _apply_lenses(self, frame: np.ndarray, skip_lenses = None) -> ProcessedFrame:
        """Apply camera settings to frame"""
        skip_lenses = skip_lenses or set()
        processed_frame = ProcessedFrame(frame)
        for lens in self.lenses:
            if lens not in skip_lenses and lens.get_enabled():
                processed_frame = lens.process(processed_frame)
        return processed_frame

    def get_snapshot(self, skip_lenses = None) -> ProcessedFrame:
        """Capture a single frame"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return self._apply_lenses(frame, skip_lenses)

    def stream(self) -> Iterator[ProcessedFrame]:
        """Stream frames from camera"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self._apply_lenses(frame)

    @staticmethod
    def list_available_cameras(max_cameras_to_check=10):
        """Check which camera indices are available on the system"""
        available_cameras = []
        for i in range(max_cameras_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras 