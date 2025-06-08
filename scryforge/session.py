from .camera import Camera
from .detector import CnnBaseDetector, DetectedCategory, Category
from .system_resources import SystemResources
from typing import Iterator, Optional, List
from dataclasses import dataclass
import numpy as np
from .lens import ScaleLens, CategoriesLense, ArucoLens, ProcessedFrame
from datetime import datetime, timedelta
import os
import json
import cv2
import imagehash
from PIL import Image
import threading
import time
import logging
from enum import Enum
from .lens import ArucoMarkerPositions
import traceback
logger = logging.getLogger(__name__)

    # when capturing training data, the base is 23px wide and 23px high

class AnnotatedScene:
    def __init__(self, image: np.ndarray, detected_categories: list[DetectedCategory]):
        self.image = image
        self.detected_categories = detected_categories
    
class CalibrationStatus(str, Enum):
    CALIBRATING = "Calibrating"
    CALIBRATED = "Calibrated"
        

class DetectionSettings:
    def __init__(self):
        self.enabled = True

class SessionSettings:
    def __init__(self):
        self.rotate_degrees: float = 0
        self.scale: float = 1.0
        self.capture_training_data: bool = False
        self.max_training_frames: int = 5000
        self.training_capture_interval: int = 2  # seconds
        self.similarity_threshold: int = 7        

class ScryForgeSession:
    def __init__(self):
        self.camera: Optional[Camera] = None
        self.detector = CnnBaseDetector()
        self.settings = SessionSettings()
        self.detection_settings = DetectionSettings()
        self.system = SystemResources()
        self.cameras = []
        
        self.aruco_lens = ArucoLens()
        self.scale_lens = ScaleLens()
        self.categories_lens = CategoriesLense()

        self.last_annotated_scene = None
        self.last_training_capture = datetime.now() - timedelta(seconds=9999)  # Initialize to a time far in the past
        self.training_frames_captured = 0
        self.last_captured_image = None
        self.run_detection_enabled = True
        self.run_detection_thread = threading.Thread(target=self.run_detection)
        self.run_detection_thread.start()

    def get_available_cameras(self) -> List[Camera]:
        """Get list of available cameras"""
        if not self.cameras:
            self.cameras = self.system.get_cameras()
        return self.cameras

    def select_camera(self, device_id: int) -> bool:
        """Select and initialize a camera"""
        camera = Camera(device_id)
        if camera.initialize():
            if self.camera:
                self.camera.release()
            self.camera = camera
            self.camera.add_lens(self.aruco_lens)
            self.camera.add_lens(self.categories_lens)  
            self.camera.add_lens(self.scale_lens)                      

            return True
        return False

    def get_selected_camera(self) -> Optional[Camera]:
        """Get the currently selected camera without side effects"""
        return self.camera

    def ensure_camera_selected(self) -> Camera:
        """Ensure a camera is selected, selecting first available if none"""
        if not self.camera:
            available_cameras = self.get_available_cameras()
            if available_cameras:
                self.select_camera(available_cameras[0].device_id)
            if not self.camera:
                raise RuntimeError("No camera available")
        return self.camera
    
    def get_snapshot(self) -> np.ndarray:
        """Get a single frame with detected bases drawn"""
        camera = self.ensure_camera_selected()
        self.categories_lens.set_enabled(self.detection_settings.enabled)
        self.scale_lens.set_scale(self.settings.scale)
        frame = camera.get_snapshot()
        return frame.image

    def stream_video(self) -> Iterator[np.ndarray]:
        """Stream frames with detected bases drawn"""
        camera = self.ensure_camera_selected()
        self.categories_lens.set_enabled(self.detection_settings.enabled)
        self.scale_lens.set_scale(self.settings.scale)
        for frame in camera.stream():
            yield frame.image 

    def get_categories(self) -> List[Category]:
        """Get list of categories"""
        return list(Category.__members__.values())

    def get_detected_categories(self) -> List[DetectedCategory]:
        """Get detected categories"""
        if not self.last_annotated_scene:
            return []
        return self.last_annotated_scene.detected_categories
    
    def get_calibration_status(self):
        """Get current calibration status and marker positions"""
        return {
            'status': CalibrationStatus.CALIBRATED if self.aruco_lens.get_calibration_status() else CalibrationStatus.CALIBRATING,
            'markers': self.aruco_lens.get_marker_positions().to_dict() if self.aruco_lens.get_calibration_status() else None
        }
    
    def reset_calibration(self):
        self.aruco_lens.reset_calibration()

    def capture_training_data(self, detected_categories: List[DetectedCategory]):
        """
        Capture training data for categories
        Args:
            detected_categories: List of DetectedCategory objects with percentage-based coordinates
        """
        camera = self.ensure_camera_selected()
        frame = camera.get_snapshot(skip_lenses=[self.scale_lens, self.categories_lens])
        
        # if the image is not aligned, skip training data capture. The labels will be wrong
        if not frame.is_aligned:
            print("Image is not aligned, skipping training data capture")
            return
        
            
        self._capture_training_frame(frame, detected_categories)

    def run_detection(self):
        while self.run_detection_enabled:
            try:
                if self.camera is None:
                    time.sleep(3)
                    continue
                    
                frame = self.camera.get_snapshot(skip_lenses=[self.scale_lens, self.categories_lens])
                detected_categories = self.detector.detect_bases(frame.image)
                if len(detected_categories) > 0:
                    self.last_annotated_scene = AnnotatedScene(frame.image, detected_categories)
                    self.categories_lens.set_detected_categories(detected_categories)
                    if self._should_capture_training_frame(frame.image):
                        self._capture_training_frame(frame, detected_categories)
                time.sleep(1)
                    
            except RuntimeError as e:
                # Camera not ready yet
                time.sleep(3)  # Wait longer on error
            except Exception as e:
                logger.error(f"Error in detection thread: {str(e)}")
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
                time.sleep(1)

    def _should_capture_training_frame(self, image: np.ndarray) -> bool:
        """Check if we should capture a training frame"""
        if not self.settings.capture_training_data:
            return False
            
        if self.training_frames_captured >= self.settings.max_training_frames:
            return False
        
        if self.last_captured_image is not None:
            if self.is_similar(self.last_captured_image, image):
                return False

        time_since_last = datetime.now() - self.last_training_capture
        return time_since_last.total_seconds() >= self.settings.training_capture_interval

    def is_similar(self, image1: np.ndarray, image2: np.ndarray, threshold: int = 5) -> bool:
        img1 = Image.fromarray(image1)
        img2 = Image.fromarray(image2)

        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        distance = hash1 - hash2
        return distance <= self.settings.similarity_threshold

    def _capture_training_frame(self, frame: ProcessedFrame, detected_categories: List[DetectedCategory]):
        """Save frame and detection data for training"""
        timestamp = int(datetime.now().timestamp() * 1000)
        image_path = f'training/images/frame_{timestamp}.jpg'
        label_path = f'training/labels/frame_{timestamp}.json'
        
        # Save image
        cv2.imwrite(image_path, frame.image)
        
        height, width = frame.image.shape[:2]
        
        label_data = {
            'data': {
                'image': f'http://localhost:5000/images/frame_{timestamp}.jpg'
            },
            'annotations': [{
                'result': [
                    {
                        'type': 'rectanglelabels',
                        'from_name': 'label',
                        'to_name': 'image',
                        'value': {
                            'x': pos.bounding_rect[0] * 100,  # Convert 0-1 to 0-100 for Label Studio
                            'y': pos.bounding_rect[1] * 100,
                            'width': pos.bounding_rect[2] * 100,
                            'height': pos.bounding_rect[3] * 100,
                            'rotation': 0,
                            'rectanglelabels': [pos.category.value]
                        }
                    } for pos in detected_categories
                ]
            }]
        }

        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)

        self.last_training_capture = datetime.now()
        self.training_frames_captured += 1
        self.last_captured_image = frame.image

    def set_calibration(self, marker_positions: ArucoMarkerPositions):
        """Set calibration from marker positions"""
        self.aruco_lens.set_marker_positions(marker_positions)




