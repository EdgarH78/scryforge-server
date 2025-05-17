from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
from .detector import Base, BaseDetector, ArucoDetector
from .detector import Category

@dataclass
class ProcessedFrame:
    image: np.ndarray
    bases: List[Base]
    scale_factor: float = 1.0  # Default to no scaling
    is_aligned: bool = False   # True if ArUco markers were used to align the image

class ImageProcessor:
    """Base class for image processors in the pipeline"""
    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        raise NotImplementedError

class RotateProcessor(ImageProcessor):
    def __init__(self, degrees: float = 0):
        self.degrees = degrees

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        if self.degrees == 0:
            return frame

        (h, w) = frame.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, self.degrees, 1.0)
        rotated_image = cv2.warpAffine(frame.image, M, (w, h))
        
        # Rotate base coordinates if they exist
        if frame.bases:
            for base in frame.bases:
                x, y = base.location
                new_x = M[0][0] * x + M[0][1] * y + M[0][2]
                new_y = M[1][0] * x + M[1][1] * y + M[1][2]
                base.location = (int(new_x), int(new_y))
                
                # Update bounding rect (simplified - could be more precise)
                x, y, w, h = base.bounding_rect
                center_x = x + w//2
                center_y = y + h//2
                new_center_x = M[0][0] * center_x + M[0][1] * center_y + M[0][2]
                new_center_y = M[1][0] * center_x + M[1][1] * center_y + M[1][2]
                base.bounding_rect = (int(new_center_x - w//2), int(new_center_y - h//2), w, h)

        return ProcessedFrame(rotated_image, frame.bases)

class DetectionProcessor(ImageProcessor):
    def __init__(self, detector: BaseDetector):
        self.detector = detector

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        bases = []
        # detect bases only if the image is aligned. It's meaningless to detect bases if the image is not aligned
        if frame.is_aligned:
            bases = self.detector.detect_bases(frame.image, frame.scale_factor)
        return ProcessedFrame(
            image=frame.image,
            bases=bases,
            scale_factor=frame.scale_factor,
            is_aligned=frame.is_aligned
        )

class DrawDetectionProcessor(ImageProcessor):
    def __init__(self, detector: BaseDetector):
        self.detector = detector
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

    def draw_detection(self, image, detection, category: Category):
        """
        Draw the detection on the image
        Args:
            image: numpy array (BGR format)
            detection: Base object with location and bounding rect
            category: Category, the color being detected
        Returns:
            image with detection drawn
        """
        if detection is not None:
            x, y, w, h = detection.bounding_rect
            # Get color from category value
            display_color = self.display_colors.get(category.value, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), display_color, 2)
            
        return image

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        for base in frame.bases:
            frame.image = self.draw_detection(frame.image, base, base.category)
        return frame

class ScaleProcessor(ImageProcessor):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        if self.scale == 1.0:
            return frame

        if frame.image is None or frame.image.size == 0:
            print("Warning: Received empty image in ScaleProcessor")
            return frame

        scaled_image = cv2.resize(frame.image, None, fx=self.scale, fy=self.scale)
        
        # Scale base coordinates
        scaled_bases = []
        for base in frame.bases:
            x, y = base.location
            new_x = int(x * self.scale)
            new_y = int(y * self.scale)
            
            x, y, w, h = base.bounding_rect
            new_rect = (
                int(x * self.scale),
                int(y * self.scale),
                int(w * self.scale),
                int(h * self.scale)
            )
            
            scaled_base = Base(
                category=base.category,
                location=(new_x, new_y),
                bounding_rect=new_rect
            )
            scaled_bases.append(scaled_base)

        return ProcessedFrame(scaled_image, scaled_bases)

class ArucoProcessor(ImageProcessor):
    def __init__(self):
        self.detector = ArucoDetector()
        self.last_valid_corners = None
        self.last_valid_ids = None
        self.consecutive_misses = 0
        self.max_misses = 1000000  

    def fallback_previous_if_possible(self) -> tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """Try to use previous valid markers if available and not too many misses"""
        self.consecutive_misses += 1
        if self.consecutive_misses >= self.max_misses:
            # Too many misses, assume markers were removed
            self.last_valid_corners = None
            self.last_valid_ids = None
            print("aruco markers were not detected for too long")
            return None, None
        
        if self.last_valid_corners is not None:
            print("using last valid aruco marker positions")
            return self.last_valid_ids, self.last_valid_corners
            
        print("aruco markers were not detected")
        return None, None

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        ids, corners = self.detector.detect_markers(frame.image)

        if len(ids) != 4:
            ids, corners = self.fallback_previous_if_possible()
            if ids is None:
                frame.is_aligned = False
                return frame
        
        self.last_valid_corners = corners
        self.last_valid_ids = ids

        # Sort corners by marker ID
        corners_by_id = sorted(zip(ids, corners), key=lambda x: x[0])

        def extract_outer_corners(corners_by_id):
            # Each marker has 4 corners in clockwise order
            # We want: top-left of TL marker, top-right of TR marker,
            # bottom-right of BR marker, bottom-left of BL marker
            tl_marker = corners_by_id[0][1][0]  # All 4 corners of TL marker
            tr_marker = corners_by_id[1][1][0]  # All 4 corners of TR marker
            br_marker = corners_by_id[2][1][0]  # All 4 corners of BR marker
            bl_marker = corners_by_id[3][1][0]  # All 4 corners of BL marker

            return (
                tl_marker[0],  # Top-left corner of top-left marker
                tr_marker[1],  # Top-right corner of top-right marker
                br_marker[2],  # Bottom-right corner of bottom-right marker
                bl_marker[3]   # Bottom-left corner of bottom-left marker
            )

        tl, tr, br, bl = extract_outer_corners(corners_by_id)
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

        # Calculate target dimensions while maintaining aspect ratio
        src_width = np.linalg.norm(tr - tl)  # Width of top edge
        src_height = np.linalg.norm(bl - tl)  # Height of left edge
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
            bases=frame.bases,
            scale_factor=target_width/max(frame.image.shape[:2]),
            is_aligned=True
        )

class Pipeline:
    def __init__(self):
        self.processors: List[ImageProcessor] = []

    def add_processor(self, processor: ImageProcessor) -> None:
        self.processors.append(processor)

    def process(self, image: np.ndarray) -> ProcessedFrame:
        frame = ProcessedFrame(image, [])
        for processor in self.processors:
            frame = processor.process(frame)
        return frame 