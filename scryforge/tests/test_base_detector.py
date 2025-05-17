import cv2
import numpy as np
import pytest
import os
from pathlib import Path
from scryforge.detector import BaseDetector, ArucoDetector, Color

@pytest.fixture
def test_image_path():
    """Fixture to provide the path to test image"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, 'resources', 'test_image_aruco.png')

@pytest.fixture
def test_image(test_image_path):
    """Fixture to load and provide the test image"""
    image = cv2.imread(test_image_path)
    if image is None:
        pytest.fail(f"Failed to load test image from {test_image_path}")
    return image

@pytest.fixture
def detector():
    """Fixture to provide a BaseDetector instance"""
    return BaseDetector()

class TestBaseDetector:
    @pytest.mark.parametrize("color,expected_coords,tolerance", [
        (Color.RED, (305, 272), 20),  # Red base coordinates
        (Color.ORANGE, (399, 164), 20)  # Orange base coordinates
    ])
    def test_base_detection(self, test_image, detector, color, expected_coords, tolerance):
        """Test that the detector can find bases at the expected coordinates"""
        # Save original image for debugging
        cv2.imwrite("debug_original.png", test_image)
        
        # Detect bases
        bases = detector.detect_bases(test_image)
        
        # Save result with detection
        result = test_image.copy()
        for base in bases:
            result = detector.draw_detection(result, base, base.color)
        cv2.imwrite("debug_result.png", result)
        
        # Find base of specified color
        color_bases = [base for base in bases if base.color == color]
        assert len(color_bases) > 0, f"No {color.value} base detected"
        
        # Get base coordinates
        base = color_bases[0]
        center_x, center_y = base.location
        expected_x, expected_y = expected_coords
        
        # Check coordinates are within tolerance
        assert abs(center_x - expected_x) <= tolerance, \
            f"{color.value} X coordinate {center_x} too far from expected {expected_x}"
        assert abs(center_y - expected_y) <= tolerance, \
            f"{color.value} Y coordinate {center_y} too far from expected {expected_y}"

    def test_detection_visualization(self, test_image, detector):
        """Test that detection can be visualized"""
        # Detect and draw
        bases = detector.detect_bases(test_image)
        
        # Draw all detected bases
        result = test_image.copy()
        for base in bases:
            result = detector.draw_detection(result, base, base.color)
        
        # Save debug image
        debug_path = "debug_detection.png"
        cv2.imwrite(debug_path, result)
        
        # Verify image was saved
        assert os.path.exists(debug_path), "Debug image was not saved"
        assert os.path.getsize(debug_path) > 0, "Debug image is empty"

