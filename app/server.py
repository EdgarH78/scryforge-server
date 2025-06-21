from flask import Flask, request, jsonify
from .detector import CnnBaseDetector, ArucoDetector
import cv2
import logging
import time
from flask_cors import CORS, cross_origin
import numpy as np
import os
import jwt 
from jwt.exceptions import InvalidTokenError
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


JWT_SECRET_CURRENT = os.environ["JWT_SECRET_CURRENT"]
JWT_SECRET_NEXT = os.environ.get("JWT_SECRET_NEXT", JWT_SECRET_CURRENT)
#use these for testing locally
#JWT_SECRET_CURRENT = "1234567890"
#JWT_SECRET_NEXT = "1234567890"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_patron_id():
    user = getattr(request, 'user', None)
    if not user or 'sub' not in user:
        return get_remote_address()
    return str(user["sub"])  # cast to string just in case

limiter = Limiter(
    app=app,
    key_func=get_patron_id,
    default_limits=[],
    headers_enabled=True  # optional but recommended
)

category_detector = CnnBaseDetector()
aruco_detector = ArucoDetector()


def verify_jwt(token: str) -> dict:
    for secret in [JWT_SECRET_CURRENT, JWT_SECRET_NEXT]:
        try:
            request.user = jwt.decode(token, secret, algorithms=["HS256"])
            return request.user
        except InvalidTokenError:
            continue
    raise InvalidTokenError("JWT verification failed with all secrets")

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header"}), 401
        token = auth_header.split(" ")[1]

        try:
            request.user = verify_jwt(token)
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/scryforge')
@limiter.request_filter
def index():
    return jsonify({'message': 'ScryForge API is running'})

@app.route('/healthz')
@limiter.request_filter
def health():
    return "OK", 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found'}), 404

@app.route('/scryforge/api/v1/image/arucolocations', methods=['POST', 'OPTIONS'])
@jwt_required
@limiter.limit("20 per second")
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
    positions = aruco_detector.get_central_coordinates(image)
    
    if positions is None:
        return jsonify({'error': 'Could not detect all markers'}), 400
        
    return jsonify({'positions': positions})

@app.route('/scryforge/api/v1/image/categories/positions', methods=['POST', 'OPTIONS'])
@jwt_required
@limiter.limit("2 per second")
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
    
    detected_categories = category_detector.detect_bases(image)
    
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
