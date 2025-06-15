from flask import Flask, request, jsonify
from .detector import CnnBaseDetector, ArucoDetector
import cv2
import logging
import time
from flask_cors import CORS, cross_origin
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/scryforge')
def index():
    return jsonify({'message': 'ScryForge API is running'})

@app.route('/healthz')
def health():
    return "OK", 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found'}), 404

@app.route('/scryforge/api/v1/image/arucolocations', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_aruco_positions():
    """POST /image/arucolocations - Get ArUco marker positions from image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400

    # Read image bytes and convert to numpy array
    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {str(e)}'}), 400
    
    # Get marker positions
    detector = ArucoDetector()
    positions = detector.get_central_coordinates(image)
    
    if positions is None:
        return jsonify({'error': 'Could not detect all markers'}), 400
        
    return jsonify({'positions': positions})

@app.route('/scryforge/api/v1/image/categories/positions', methods=['POST', 'OPTIONS'])
@cross_origin()
def process_category_positions():
    """POST /image/categories/positions - Detect categories in uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400

    # Read image bytes and convert to numpy array
    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {str(e)}'}), 400
    
    # Detect categories in image
    detector = CnnBaseDetector()
    detected_categories = detector.detect_bases(image)
    
    # Convert to pixel coordinates
    height, width = image.shape[:2]
    pixel_positions = [{
        'category': pos.category.value,
        'x': int(pos.location[0] * width),
        'y': int(pos.location[1] * height),
        'width': int(pos.bounding_rect[2] * width),
        'height': int(pos.bounding_rect[3] * height)
    } for pos in detected_categories]
    
    return jsonify({'positions': pixel_positions})


def profile_timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper
