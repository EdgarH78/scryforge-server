import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict
from abc import ABC, abstractmethod

class Category(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    VIOLET = "violet"
    YELLOW = "yellow"
    ORANGE = "orange"
    TURQUOISE = "turquoise"
    PINK = "pink"
    WHITE = "white"
    BLACK = "black"
    LIME_GREEN_VIOLET = "lime_green_violet"
    PINK_GREEN = "pink_green"
    YELLOW_NIGHT_BLUE = "yellow_night_blue"
    LIGHT_BLUE_ORANGE = "light_blue_orange"
    BROWN_TURQUOISE = "brown_turquoise"
    GIANT_RED_OCTOPUS = "giant_red_octopus"
    TREANT = "treant"
    ANCIENT_GOLD_DRAGON = "ancient_gold_dragon"
    ANCIENT_SILVER_DRAGON = "ancient_silver_dragon"
    ANCIENT_BLUE_DRAGON = "ancient_blue_dragon"
    ANCIENT_GREEN_DRAGON = "ancient_green_dragon"
    BLUE_DRACO_LICH = "blue_dragolich"
    ANCIENT_RED_DRAGON = "ancient_red_dragon"
    ANCIENT_WHITE_DRAGON = "ancient_white_dragon"
    ANCIENT_BLACK_DRAGON = "ancient_black_dragon"
    ANCIENT_GREY_DRAGON = "ancient_grey_dragon"
    ANCIENT_PURPLE_DRAGON = "ancient_purple_dragon"

    @classmethod
    def classes(cls):
        return ["__background__"] + [c.value for c in cls]

    @classmethod
    def from_string(cls, value: str):
        """Get Category enum from string value, case-insensitive"""
        value = value.lower()
        for cat in cls:
            if cat.value.lower() == value:
                return cat
        return None

@dataclass
class Base:
    category: Category  # Renamed from color
    location: Tuple[int, int]  # center x, y
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h


class Detector:
    def __init__(self):
        """Initialize the detector with a pre-trained model"""
        # Load pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.model.eval()  # Set to evaluation mode
        
        # Get the COCO category names
        self.categories = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def detect(self, image, confidence_threshold=0.5):
        """
        Detect objects in the image
        Args:
            image: numpy array (BGR format from OpenCV)
            confidence_threshold: minimum confidence score to include a detection
        Returns:
            list of dictionaries containing detections
            each dict has 'label', 'confidence', and 'bbox' (x1,y1,x2,y2)
        """
        # Convert BGR to RGB
        image_rgb = image[..., ::-1]
        
        # Convert to torch tensor (normalize)
        image_tensor = F.to_tensor(image_rgb)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        detections = []
        pred = predictions[0]  # First (and only) image in batch
        
        for idx in range(len(pred['boxes'])):
            score = pred['scores'][idx].item()
            if score > confidence_threshold:
                label = self.categories[pred['labels'][idx].item()]
                bbox = pred['boxes'][idx].cpu().numpy()
                
                detections.append({
                    'label': label,
                    'confidence': score,
                    'bbox': bbox.astype(np.int32)  # Convert to integer coordinates
                })
        
        return detections 

class SilhouetteDetector:
    def __init__(self):
        """Initialize SAM model for silhouette detection"""
        model_type = "vit_h"
        checkpoint = "sam_vit_h_4b8939.pth"
        
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"SAM model checkpoint not found at {checkpoint}. "
                "Please download it from: "
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def detect_silhouette(self, image):
        """
        Detect object silhouettes in image
        Args:
            image: numpy array (BGR format from OpenCV)
        Returns:
            masks: binary masks for detected objects
        """
        self.predictor.set_image(image)
        
        # Get automatic mask predictions
        masks, scores, _ = self.predictor.predict()
        
        return masks, scores 

class ContourDetector:
    def __init__(self):
        pass

    def detect_silhouette(self, image, threshold=127):
        """
        Detect silhouettes using contour detection
        Args:
            image: numpy array (BGR format from OpenCV)
            threshold: threshold for binary image conversion
        Returns:
            contours: list of detected contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours 

class BaseDetector(ABC):
    @abstractmethod
    def detect_bases(self, image: np.ndarray, scale_factor: float = 1.0) -> list[Base]:
        """Detect all bases in the image"""
        pass

class ArucoDetector:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_markers(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect ArUco markers in the image
        Returns:
            Tuple of (marker_ids, marker_corners)
            marker_ids: numpy array of detected marker IDs
            marker_corners: List of corner coordinates for each marker
        """
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            return np.array([]), []
        return ids.flatten(), corners

    def draw_markers(self, image: np.ndarray, ids: List[int], corners: List[np.ndarray]) -> np.ndarray:
        if len(ids) == 0:
            return image
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return image

class CnnBaseDetector(BaseDetector):
    def __init__(self, model_path="fasterrcnn_token_detector.pth"):
        self.bases = []
        self.last_image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, len(Category.classes())
        )
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        
        # Class mapping
        self.classes = Category.classes()
        self.category_colors = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "violet": (238, 130, 238),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "turquoise": (208, 224, 64),
            "pink": (203, 192, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "lime_green_violet": (153, 255, 204),        # Pale lime green
            "pink_green": (180, 255, 180),               # Pastel mix
            "yellow_night_blue": (140, 140, 255),        # Dimmed blue/yellow mix
            "light_blue_orange": (100, 150, 255),        # Soft orange tint
            "brown_turquoise": (139, 87, 66),            # Brownish
            "giant_red_octopus": (64, 0, 128),           # Deep magenta
            "treant": (34, 139, 34),                     # Forest green
            "ancient_gold_dragon": (0, 215, 255),        # Bright gold (BGR)
            "ancient_silver_dragon": (192, 192, 192),
            "ancient_blue_dragon": (255, 100, 100),
            "ancient_green_dragon": (0, 180, 0),
            "blue_dragolich": (153, 50, 204),            # Dark orchid
            "ancient_red_dragon": (0, 0, 180),
            "ancient_white_dragon": (245, 245, 245),
            "ancient_black_dragon": (30, 30, 30),
            "ancient_grey_dragon": (128, 128, 128),
            "ancient_purple_dragon": (128, 0, 128)
        }

        self.run_detection_enabled = True
        self.run_detection_thread = threading.Thread(target=self.run_base_detection)
        self.run_detection_thread.start()

    def detect_bases(self, image: np.ndarray, scale_factor: float = 1.0) -> list[Base]:
        """Detect bases using the trained CNN model"""
        self.last_image = image
        return self.bases
        
    def run_base_detection(self):
        while self.run_detection_enabled:
            time.sleep(3)

            if self.last_image is None:
                continue

            image_rgb = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(image_tensor)

            pred = predictions[0]
            threshold = 0.5

            best_by_category = {}

            for idx in range(len(pred['boxes'])):
                score = pred['scores'][idx].item()
                if score < threshold:
                    continue

                label_idx = pred['labels'][idx].item()
                class_name = self.classes[label_idx]

                category = Category.from_string(class_name)
                if category is None:
                    continue

                if category not in best_by_category or score > best_by_category[category][0]:
                    box = pred['boxes'][idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    best_by_category[category] = (
                        score,
                        Base(
                            category=category,
                            location=center,
                            bounding_rect=(x1, y1, x2 - x1, y2 - y1)
                        )
                    )

            self.bases = [b for _, b in best_by_category.values()]

        
