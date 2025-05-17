import cv2
import numpy as np
import pytest
import os
from scryforge.detector import ArucoDetector

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
    """Fixture to provide an ArucoDetector instance"""
    return ArucoDetector()

class TestArucoDetector:
    def test_detector_initialization(self, detector):
        """Test that detector initializes properly"""
        assert detector.aruco_dict is not None
        assert detector.aruco_params is not None
        assert detector.detector is not None

    def test_detect_four_markers(self, test_image, detector):
        """Test that all four ArUco markers are detected"""
        ids, corners = detector.detect_markers(test_image)
        
        # Save debug image
        debug_image = test_image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
        cv2.imwrite("debug_aruco.png", debug_image)
        
        # Check we found exactly 4 markers
        assert len(ids) == 4, f"Expected 4 markers, found {len(ids)}"
        assert len(corners) == 4, f"Expected 4 corners, found {len(corners)}"
        
        # Check we found the correct IDs (0, 1, 2, 3)
        expected_ids = set([0, 1, 2, 3])
        found_ids = set(ids)
        assert found_ids == expected_ids, f"Expected IDs {expected_ids}, found {found_ids}"

    def test_marker_positions(self, test_image, detector):
        """Test that markers are in correct positions relative to each other"""
        ids, corners = detector.detect_markers(test_image)
        
        # Sort corners by ID
        corners_by_id = sorted(zip(ids, corners))
        points = np.array([corner[0].mean(axis=0) for _, corner in corners_by_id])
        
        # Extract points
        tl, tr, br, bl = points  # ID 0=TL, 1=TR, 2=BR, 3=BL
        
        # Check relative positions
        assert tl[1] < bl[1], "Top-left Y should be less than bottom-left Y"
        assert tr[1] < br[1], "Top-right Y should be less than bottom-right Y"
        assert tl[0] < tr[0], "Top-left X should be less than top-right X"
        assert bl[0] < br[0], "Bottom-left X should be less than bottom-right X" 