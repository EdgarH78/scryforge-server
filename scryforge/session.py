from .camera import Camera, CameraSettings
from .detector import CnnBaseDetector, Base, Category
from .system_resources import SystemResources
from typing import Iterator, Optional, List
from dataclasses import dataclass
import numpy as np
from .pipeline import Pipeline, RotateProcessor, DetectionProcessor, ScaleProcessor, DrawDetectionProcessor, ArucoProcessor, ProcessedFrame
from datetime import datetime, timedelta
import os
import json
import cv2
import imagehash
from PIL import Image

    # when capturing training data, the base is 23px wide and 23px high

@dataclass
class CategoryPosition:
    category: Category
    x: float  # percentage (0-100)
    y: float  # percentage (0-100)
    width: float = 0  # percentage (0-100)
    height: float = 0  # percentage (0-100)

class AnnotatedScene:
    def __init__(self, image: np.ndarray, bases: list[Base]):
        self.image = image
        self.bases = bases

class DetectionSettings:
    def __init__(self):
        self.enabled = True

class SessionSettings:
    def __init__(self):
        self.rotate_degrees: float = 0
        self.scale: float = 1.0
        self.capture_training_data: bool = False
        self.max_training_frames: int = 5000
        self.training_capture_interval: int = 30  # seconds
        self.similarity_threshold: int = 7        

class ScryForgeSession:
    def __init__(self):
        self.camera: Optional[Camera] = None
        self.detector = CnnBaseDetector()
        self.settings = SessionSettings()
        self.detection_settings = DetectionSettings()
        self.system = SystemResources()
        self.cameras = []
        
        self.aruco_processor = ArucoProcessor()
        self.detection_processor = DetectionProcessor(self.detector)
        self.draw_detection_processor = DrawDetectionProcessor(self.detector)

        self.last_training_capture = datetime.now() - timedelta(seconds=9999)  # Initialize to a time far in the past
        self.training_frames_captured = 0
        self.last_captured_image = None

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
            # Only pass flip setting to camera
            self.camera.settings.is_flipped = False  # Reset on new camera
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
        frame = camera.get_snapshot()
        return self._process_frame(frame).image

    def stream_video(self) -> Iterator[np.ndarray]:
        """Stream frames with detected bases drawn"""
        camera = self.ensure_camera_selected()
        for frame in camera.stream():
            yield self._process_frame(frame).image 

    def get_categories(self) -> List[Category]:
        """Get list of categories"""
        return list(Category.__members__.values())

    def get_annotated_scene(self) -> ProcessedFrame:
        """Get detected bases"""
        camera = self.ensure_camera_selected()
        frame = camera.get_snapshot()
        pipeLine = Pipeline()
        pipeLine.add_processor(ArucoProcessor())
        pipeLine.add_processor(DetectionProcessor(self.detector))
        processed = pipeLine.process(frame)
        if self._should_capture_training_frame(processed.image):
            self._capture_training_frame(processed)
        return processed
    
    def capture_training_data(self, positions: List[CategoryPosition]):
        """
        Capture training data for categories with their positions
        Args:
            image: The image to annotate
            positions: List of CategoryPosition with x,y as percentages (0-100)
        """
        camera = self.ensure_camera_selected()
        image = camera.get_snapshot()
        pipeLine = Pipeline()
        pipeLine.add_processor(ArucoProcessor())
        processed = pipeLine.process(image)
        
        # if the image is not aligned, skip training data capture. The labels will be wrong
        if not processed.is_aligned:
            print("Image is not aligned, skipping training data capture")
            return
        
        height, width = processed.image.shape[:2]
        bases = []
        
        for pos in positions:
            x = int((pos.x / 100) * width)
            y = int((pos.y / 100) * height)
            bases.append(Base(
                category=pos.category,
                location=(x, y),
                bounding_rect=(x-11, y-11, 23, 23)
            ))
            
        self._capture_training_frame(ProcessedFrame(processed.image, bases))

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

    def _capture_training_frame(self, frame: ProcessedFrame):
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
                            'x': (base.bounding_rect[0] / width) * 100,
                            'y': (base.bounding_rect[1] / height) * 100,
                            'width': (base.bounding_rect[2] / width) * 100,
                            'height': (base.bounding_rect[3] / height) * 100,
                            'rotation': 0,
                            'rectanglelabels': [base.category.value]
                        }
                    } for base in frame.bases
                ]
            }]
        }

        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)

        self.last_training_capture = datetime.now()
        self.training_frames_captured += 1
        self.last_captured_image = frame.image


    def _process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        pipeLine = Pipeline()
        pipeLine.add_processor(RotateProcessor(self.settings.rotate_degrees))
        if self.detection_settings.enabled:
            pipeLine.add_processor(self.aruco_processor)
            pipeLine.add_processor(self.detection_processor)
            # Check if we should capture this frame for training
            processed_for_capture = pipeLine.process(frame)
            if self._should_capture_training_frame(processed_for_capture.image):                
                self._capture_training_frame(processed_for_capture)
            pipeLine.add_processor(self.draw_detection_processor)
        pipeLine.add_processor(ScaleProcessor(self.settings.scale))
        processed = pipeLine.process(frame)
        return processed



