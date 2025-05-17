import cv2
import logging
import time
import numpy as np
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

class CameraSettings:
    def __init__(self):
        self.is_flipped: bool = False

class Camera:
    _instances = {}  # Class variable to store camera instances

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.settings = CameraSettings()
        self.max_retries = 3

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

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _apply_settings(self, frame: np.ndarray) -> np.ndarray:
        """Apply camera settings to frame"""
        if self.settings.is_flipped:
            frame = cv2.flip(frame, -1)
        return frame

    def get_snapshot(self) -> np.ndarray:
        """Capture a single frame"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return self._apply_settings(frame)

    def stream(self) -> Iterator[np.ndarray]:
        """Stream frames from camera"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self._apply_settings(frame)

    def snap_frame(self):
        """Capture a single frame from the camera"""
        for attempt in range(self.max_retries):
            if not self.cap or not self.cap.isOpened():
                if not self.initialize():
                    logger.error(f"Failed to initialize camera on attempt {attempt + 1}")
                    continue

            try:
                start = time.time()
                ret, frame = self.cap.read()
                logger.info(f"Camera read took {time.time() - start:.3f} seconds")
                
                if ret:
                    return self._apply_settings(frame)
                else:
                    logger.warning(f"Failed to capture frame on attempt {attempt + 1}")
                    self.initialize()  # Try to reinitialize
            except Exception as e:
                logger.error(f"Error capturing frame: {str(e)}")
                self.initialize()  # Try to reinitialize

        raise RuntimeError(f"Failed to capture frame after {self.max_retries} attempts")

    def stream_frames(self):
        """Generator that yields frames continuously from the camera"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera is not initialized. Use with-statement to handle camera lifecycle.")
            
        consecutive_failures = 0
        max_failures = 5  # Maximum number of consecutive failures before giving up
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"Failed to read from camera {self.device_id} after {max_failures} attempts")
                        break
                    logger.warning(f"Failed to read frame from camera {self.device_id}, attempt {consecutive_failures}")
                    continue
                
                consecutive_failures = 0  # Reset counter on successful read
                yield self._apply_settings(frame)
                
            except Exception as e:
                logger.error(f"Error reading from camera {self.device_id}: {str(e)}")
                break

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