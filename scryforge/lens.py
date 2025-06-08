from dataclasses import dataclass
from typing import List
import numpy as np
import cv2
from .detector import ArucoDetector
from .detector import DetectedCategory
from typing import Tuple

@dataclass
class ArucoMarkerPositions:
    """Store ArUco marker positions for calibration"""
    top_left: Tuple[float, float]     # ID 0
    top_right: Tuple[float, float]    # ID 1
    bottom_right: Tuple[float, float] # ID 2
    bottom_left: Tuple[float, float]  # ID 3
    
    def to_dict(self):
        return {
            'top_left': self.top_left,
            'top_right': self.top_right,
            'bottom_right': self.bottom_right,
            'bottom_left': self.bottom_left
        }

@dataclass
class ProcessedFrame:
    image: np.ndarray
    scale_factor: float = 1.0  # Default to no scaling
    is_aligned: bool = False   # True if ArUco markers were used to align the image

class Lens:
    """Base class for image processors in the pipeline"""
    def __init__(self):
        self.enabled = True
    
    def get_enabled(self) -> bool:
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        raise NotImplementedError


class CategoriesLense(Lens):
    def __init__(self):
        super().__init__()
        self.detected_categories = []
        # Add display colors
        self.display_colors = {
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

    def draw_detection(self, image, detected: DetectedCategory):
        """Draw the detection on the image using pixel coordinates"""
        if detected is not None:
            # Convert percentages back to pixels for this image
            _, rect = detected.to_pixels(image.shape[1], image.shape[0])
            x, y, w, h = rect
            
            # Get color from category value
            display_color = self.display_colors.get(detected.category.value, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), display_color, 2)
            
        return image

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        for detected in self.detected_categories:
            frame.image = self.draw_detection(frame.image, detected)
        return frame
    
    def set_detected_categories(self, detected_categories: List[DetectedCategory]):
        self.detected_categories = detected_categories

class ScaleLens(Lens):  
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def set_scale(self, scale: float):
        self.scale = scale

    def get_scale(self) -> float:
        return self.scale   

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        if self.scale == 1.0:
            return frame

        if frame.image is None or frame.image.size == 0:
            print("Warning: Received empty image in ScaleProcessor")
            return frame

        scaled_image = cv2.resize(frame.image, None, fx=self.scale, fy=self.scale)
        
        return ProcessedFrame(scaled_image, scale_factor=self.scale, is_aligned=frame.is_aligned)

class ArucoLens(Lens):
    def __init__(self):
        super().__init__()
        self.detector = ArucoDetector()
        self.corners = None  # Store just the 4 outer corners we need
    
    def reset_calibration(self):
        self.corners = None
    
    def get_calibration_status(self) -> bool:
        return self.corners is not None

    def set_marker_positions(self, positions: ArucoMarkerPositions):
        """Set calibration from marker positions"""
        self.corners = np.array([
            positions.top_left,      # Top-left corner
            positions.top_right,     # Top-right corner
            positions.bottom_right,  # Bottom-right corner
            positions.bottom_left    # Bottom-left corner
        ], dtype=np.float32)

    def get_marker_positions(self) -> ArucoMarkerPositions:
        """Get current marker positions"""
        if not self.get_calibration_status():
            raise ValueError("Not calibrated")
            
        return ArucoMarkerPositions(
            top_left=tuple(float(x) for x in self.corners[0]),
            top_right=tuple(float(x) for x in self.corners[1]),
            bottom_right=tuple(float(x) for x in self.corners[2]),
            bottom_left=tuple(float(x) for x in self.corners[3])
        )

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        if self.corners is not None:
            src_pts = self.corners
        else:
            corners = self.detector.get_outer_corners(frame.image)
            if len(corners) != 4:
                frame.is_aligned = False
                return frame

            src_pts = np.array([
                corners['top_left'],  # Top-left corner of top-left marker
                corners['top_right'],  # Top-right corner of top-right marker
                corners['botom_right'],  # Bottom-right corner of bottom-right marker
                corners['bottom_left']   # Bottom-left corner of bottom-left marker
            ], dtype=np.float32)
            
            self.corners = src_pts

        # Calculate target dimensions while maintaining aspect ratio
        src_width = np.linalg.norm(src_pts[1] - src_pts[0])  # Width of top edge
        src_height = np.linalg.norm(src_pts[3] - src_pts[0])  # Height of left edge
        aspect_ratio = src_height / src_width
        
        # Calculate target width to maintain maximum resolution
        # Use the larger of width/height to determine scale
        max_dimension = max(src_width, src_height)
        scale = max_dimension / max(frame.image.shape[:2])
        target_width = int(src_width * scale)
        target_height = int(target_width * aspect_ratio)
        
        # Ensure minimum size of 300px
        if target_width < 300:
            target_width = 300
            target_height = int(target_width * aspect_ratio)

        # Define target points preserving aspect ratio
        dst_pts = np.array([
            [0, 0],                      # Top-left
            [target_width, 0],           # Top-right
            [target_width, target_height], # Bottom-right
            [0, target_height]           # Bottom-left
        ], dtype=np.float32)

        # Calculate perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Use CUBIC interpolation for better quality
        warped = cv2.warpPerspective(
            frame.image, M, (target_width, target_height),
            flags=cv2.INTER_CUBIC
        )

        # Debug: Draw the source and destination points
        debug_img = frame.image.copy()
        for pt in src_pts:
            cv2.circle(debug_img, tuple(map(int, pt)), 5, (0, 0, 255), -1)
        cv2.imwrite('debug_source_points.png', debug_img)
        cv2.imwrite('debug_warped.png', warped)

        frame.is_aligned = True
        return ProcessedFrame(
            image=warped,
            scale_factor=target_width/max(frame.image.shape[:2]),
            is_aligned=True
        )