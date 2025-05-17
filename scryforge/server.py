from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from .session import ScryForgeSession, CategoryPosition
from .detector import Category
import cv2
import logging
import time
import base64
import os
import xml.etree.ElementTree as ET
import json
from flask_cors import CORS
from dataclasses import dataclass, asdict


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
    if settings.get('is_flipped') is not None:
        get_session().get_selected_camera().settings.is_flipped = settings['is_flipped']
    if settings.get('rotate_degrees') is not None:
        get_session().settings.rotate_degrees = float(settings['rotate_degrees'])
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
        'is_flipped': get_session().camera.settings.is_flipped,
        'rotate_degrees': get_session().settings.rotate_degrees,
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
    annotated_scene = get_session().get_annotated_scene()
    frame_height, frame_width = annotated_scene.image.shape[:2]
    
    positions = [
        {
            'category': base.category.value,  # Convert enum to string
            'x': (base.location[0] / frame_width) * 100,
            'y': (base.location[1] / frame_height) * 100,
            'width': (base.bounding_rect[2] / frame_width) * 100,
            'height': (base.bounding_rect[3] / frame_height) * 100
        } for base in annotated_scene.bases
    ]

    return jsonify({'positions': positions})

@app.route('/categories/positions', methods=['POST'])
def capture_categories():
    """POST /categories/positions"""
    try:
        positions_data = request.json.get('positions', [])
        positions = [CategoryPosition(
            category=Category(pos['category']),  # Convert string back to enum
            x=float(pos['x']),
            y=float(pos['y']),
            width=float(pos.get('width', 0)),
            height=float(pos.get('height', 0))
        ) for pos in positions_data]
        get_session().capture_training_data(positions)
        return "Training data captured", 200
    except Exception as e:
        return str(e), 400

def profile_timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper
