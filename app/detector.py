import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
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
class DetectedCategory:
    category: Category
    location: Tuple[float, float]  # x,y as percentages (0-1) of image width/height
    bounding_rect: Tuple[float, float, float, float]  # x,y,w,h as percentages of image dimensions

    @classmethod
    def from_pixels(cls, category: Category, location: Tuple[int, int], rect: Tuple[int, int, int, int], image_width: int, image_height: int) -> 'DetectedCategory':
        """Create from pixel coordinates"""
        return cls(
            category=category,
            location=(
                (location[0] / image_width),
                (location[1] / image_height)
            ),
            bounding_rect=(
                (rect[0] / image_width),
                (rect[1] / image_height),
                (rect[2] / image_width),
                (rect[3] / image_height)
            )
        )

    def to_pixels(self, image_width: int, image_height: int) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """Convert percentages back to pixel coordinates"""
        return (
            (
                int((self.location[0]) * image_width),
                int((self.location[1]) * image_height)
            ),
            (
                int((self.bounding_rect[0]) * image_width),
                int((self.bounding_rect[1]) * image_height),
                int((self.bounding_rect[2]) * image_width),
                int((self.bounding_rect[3]) * image_height)
            )
        )

    def to_dict(self):
        return {
            'category': self.category.value,
            'x': self.location[0],
            'y': self.location[1],
            'width': self.bounding_rect[2],
            'height': self.bounding_rect[3]
        }

@dataclass
class Alignment:
    observed_bases: List[DetectedCategory]
    expected_bases: List[DetectedCategory]



class BaseDetector(ABC):
    @abstractmethod
    def detect_bases(self, image: np.ndarray) -> list[DetectedCategory]:
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

    def get_central_coordinates(self, image: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
        ids, corners = self.detect_markers(image)
        if len(ids) == 0:
            return {}

        result = {}

        position_map = {
            0: 'top_left',
            1: 'top_right', 
            2: 'bottom_right',
            3: 'bottom_left'
        }
        
        for marker_id, marker_corners in zip(ids, corners):
            if marker_id not in position_map:
                continue

            position = position_map[marker_id]

            # Compute center of the marker from all 4 corners
            center = np.mean(marker_corners[0], axis=0)  # shape (4, 2)
            result[position] = (float(center[0]), float(center[1]))

        return result if result else None
        

class CnnBaseDetector(BaseDetector):
    def __init__(self, model_path="fasterrcnn_token_detector.pth"):
        self.bases = []
        self.last_image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alignment = None
        
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

    def detect_bases(self, image: np.ndarray) -> list[DetectedCategory]:
        """Detect bases and apply alignment correction if available"""
        detected = self._detect_raw(image)
        
        if self.alignment is not None:
            # Apply transformation to detected locations
            for d in detected:
                # Transform center location (shape: 1,1,2)
                loc = np.array([[d.location]], dtype=np.float32)
                corrected_loc = cv2.transform(loc, self.alignment)[0][0]
                d.location = tuple(corrected_loc)
                
                # Transform both corners of bounding box
                x, y, w, h = d.bounding_rect
                top_left = np.array([[[x, y]]], dtype=np.float32)
                bottom_right = np.array([[[x + w, y + h]]], dtype=np.float32)
                
                tl = cv2.transform(top_left, self.alignment)[0][0]
                br = cv2.transform(bottom_right, self.alignment)[0][0]
                
                new_w = br[0] - tl[0]
                new_h = br[1] - tl[1]
                d.bounding_rect = (tl[0]+ (new_w/2), tl[1]+ (new_h/2), new_w, new_h)
                
        return detected

    def _detect_raw(self, image: np.ndarray) -> list[DetectedCategory]:
        """Detect bases using the trained CNN model"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                    DetectedCategory.from_pixels(
                        category=category,
                        location=center,
                        rect=(x1, y1, x2 - x1, y2 - y1),
                        image_width=image.shape[1],
                        image_height=image.shape[0]
                    )
                )

        return [b for _, b in best_by_category.values()]

    def align(self, alignment: Alignment):
        """Store alignment data to correct future detections"""
        if len(alignment.observed_bases) < 2 or len(alignment.expected_bases) < 2:
            return  # Need at least 2 points for alignment
        
        # Convert percentage coordinates to points for transformation calculation
        observed_points = np.array([d.location for d in alignment.observed_bases], dtype=np.float32)
        expected_points = np.array([d.location for d in alignment.expected_bases], dtype=np.float32)
        
        # Calculate transformation matrix
        matrix, _ = cv2.estimateAffinePartial2D(observed_points, expected_points)
        if matrix is not None:
            self.alignment = matrix
            self.alignment_data = {
                "matrix": matrix.tolist(),
                "observed": observed_points.tolist(),
                "expected": expected_points.tolist()
            }


        
