from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from .session import ScryForgeSession
from .detector import Category, DetectedCategory, ArucoDetector, CnnBaseDetector
from .lens import ArucoMarkerPositions
import cv2
import logging
import time
import base64
import os
import xml.etree.ElementTree as ET
import json
from flask_cors import CORS
from dataclasses import dataclass, asdict
import numpy as np


IMAGE_DIR = os.path.join(os.getcwd(), "training", "images")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable for all routes
session = None

def get_session():
    global session
    if session is None:
        session = ScryForgeSession()
    return session

# Create training directories
os.makedirs('training/images', exist_ok=True)
os.makedirs('training/labels', exist_ok=True)

@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve training images"""
    try:
        return send_from_directory(IMAGE_DIR, filename, mimetype='image/jpeg')
    except FileNotFoundError:
        return "Image not found", 404

@app.route('/')
def index():
    """GET /"""
    cameras = get_session().get_available_cameras()
    return render_template('index.html',
                         cameras=cameras,
                         selected_camera=get_session().camera)

@app.route('/get_snapshot')
def get_snapshot():
    """GET /get_snapshot"""
    try:
        frame = get_session().get_snapshot()
        ret, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()
        return Response(frame_bytes, mimetype='image/png')
    except RuntimeError as e:
        return str(e), 400

@app.route('/stream_video')
def stream_video():
    """GET /stream_video"""
    def gen_frames():
        try:
            for frame in get_session().stream_video():
                ret, buffer = cv2.imencode('.png', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')
        except RuntimeError as e:
            yield str(e).encode()

    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_cameras')
def get_cameras():
    """GET /get_cameras"""
    cameras = get_session().get_available_cameras()
    return jsonify([{
        'id': camera.device_id,
        'name': f'Camera {camera.device_id}'
    } for camera in cameras])

@app.route('/select_camera', methods=['POST'])
def select_camera():
    """POST /select_camera"""
    camera_id = request.json.get('camera_id')
    if camera_id is None:
        return "Missing camera_id", 400
    
    success = get_session().select_camera(int(camera_id))
    if success:
        return "Camera selected", 200
    return "Failed to select camera", 400

@app.route('/camera/settings', methods=['POST'])
def update_camera_settings():
    """POST /camera/settings"""
    if not get_session().camera:
        return "No camera selected", 400
        
    settings = request.json

    if settings.get('scale') is not None:
        get_session().settings.scale = float(settings['scale'])
    if settings.get('capture_training') is not None:
        get_session().settings.capture_training_data = settings['capture_training']
        if settings['capture_training']:
            get_session().training_frames_captured = 0  # Reset counter when enabling
        
    return "Settings updated", 200

@app.route('/camera/settings', methods=['GET'])
def get_camera_settings():
    """GET /camera/settings"""
    if not get_session().camera:
        return "No camera selected", 400
        
    return jsonify({
        'is_flipped': False,
        'rotate_degrees': 0,
        'scale': get_session().settings.scale
    })

@app.route('/detection/settings', methods=['GET'])
def get_detection_settings():
    """GET /detection/settings"""
    return jsonify({
        'enabled': get_session().detection_settings.enabled
    })

@app.route('/detection/settings', methods=['POST'])
def update_detection_settings():
    """POST /detection/settings"""
    settings = request.json
    if settings.get('enabled') is not None:
        get_session().detection_settings.enabled = settings['enabled']
    return "Settings updated", 200

@app.route('/categories')
def get_categories():
    """GET /categories"""
    categories = get_session().get_categories()
    output = {
        'categories': [cat.value for cat in categories]  # Convert enum to string values
    }
    return jsonify(output)

@app.route('/categories/positions')
def get_categories_positions():
    """GET /get_annotated_scene"""
    positions = get_session().get_detected_categories()
    return jsonify({'positions': [pos.to_dict() for pos in positions]})

@app.route('/categories/positions', methods=['POST'])
def capture_categories():
    """POST /categories/positions"""
    try:
        positions_data = request.json.get('positions', [])
        detected_categories = [DetectedCategory(
            category=Category(pos['category']),
            location=(float(pos['x']), float(pos['y'])),  # Need to check if UI sends 0-100 or 0-1
            bounding_rect=(float(pos['x']), float(pos['y']), 
                          float(pos.get('width', 0)), float(pos.get('height', 0)))
        ) for pos in positions_data]
        get_session().capture_training_data(detected_categories)
        return "Training data captured", 200
    except Exception as e:
        return str(e), 400

@app.route('/calibration')
def get_calibration_status():
    """GET /calibration"""
    return jsonify(get_session().get_calibration_status())

@app.route('/calibration', methods=['POST'])
def set_calibration():
    """POST /calibration"""
    try:
        calibration_data = request.json
        if 'markers' not in calibration_data:
            return jsonify({'error': 'Missing markers in calibration data'}), 400
            
        markers = calibration_data['markers']
        marker_positions = ArucoMarkerPositions(
            top_left=tuple(markers['top_left']),
            top_right=tuple(markers['top_right']),
            bottom_right=tuple(markers['bottom_right']),
            bottom_left=tuple(markers['bottom_left'])
        )
        
        get_session().set_calibration(marker_positions)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calibration', methods=['DELETE'])
def reset_calibration():
    """Reset ArUco marker calibration"""
    try:
        get_session().reset_calibration()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/category/alignment', methods=['POST'])
def set_category_alignment():
    """POST /calibration/status"""



@app.route('/image/arucolocations', methods=['POST'])
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

@app.route('/image/categories/positions', methods=['POST'])
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
