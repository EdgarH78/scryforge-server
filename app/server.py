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
import threading


JWT_SECRET_CURRENT = os.environ.get("JWT_SECRET_CURRENT")
JWT_SECRET_NEXT = os.environ.get("JWT_SECRET_NEXT")
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

# Log JWT secret status
if not JWT_SECRET_CURRENT or JWT_SECRET_CURRENT.strip() == "":
    logger.error("❌ JWT_SECRET_CURRENT is empty or not set!")
else:
    logger.info("✅ JWT_SECRET_CURRENT is set")

if not JWT_SECRET_NEXT or JWT_SECRET_NEXT.strip() == "":
    logger.warning("⚠️ JWT_SECRET_NEXT is empty or not set, using JWT_SECRET_CURRENT")
else:
    logger.info("✅ JWT_SECRET_NEXT is set")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route('/scryforge/api/v1/image/arucolocations', methods=["OPTIONS"])
@app.route('/scryforge/api/v1/image/categories/positions', methods=["OPTIONS"])
def preflight():
    return '', 204

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

category_detector = None
aruco_detector = ArucoDetector()
models_ready = False

def load_models():
    global category_detector, aruco_detector, models_ready
    try:
        category_detector = CnnBaseDetector()
        models_ready = True
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.exception(f"❌ Failed to load models: {e}")

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()        

def verify_jwt(token: str) -> dict:
    logger.info("🔐 Attempting JWT verification")
    
    for i, secret in enumerate([JWT_SECRET_CURRENT, JWT_SECRET_NEXT]):
        try:
            logger.info("🔑 Trying secret %d (current: %s)", i+1, "current" if i == 0 else "next")
            request.user = jwt.decode(token, secret, algorithms=["HS256"])
            logger.info("✅ JWT verification successful with secret %d", i+1)
            return request.user
        except InvalidTokenError as e:
            logger.warning("⚠️ JWT verification failed with secret %d: %s", i+1, str(e))
            continue
        except Exception as e:
            logger.error("❌ Unexpected error during JWT verification with secret %d: %s", i+1, str(e))
            continue
    
    logger.error("❌ JWT verification failed with all secrets")
    raise InvalidTokenError("JWT verification failed with all secrets")

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info("🔒 JWT authentication required for endpoint")
        
        auth_header = request.headers.get("Authorization", "")
        logger.info("📋 Authorization header present: %s", "yes" if auth_header else "no")
        
        if not auth_header.startswith("Bearer "):
            logger.error("❌ Missing or malformed Authorization header: %s", auth_header[:20] + "..." if len(auth_header) > 20 else auth_header)
            return jsonify({"error": "Missing or malformed Authorization header"}), 401
        
        token = auth_header.split(" ")[1]
        logger.info("🎫 JWT token extracted (length: %d)", len(token))

        try:
            logger.info("🔐 Calling verify_jwt...")
            request.user = verify_jwt(token)
            logger.info("✅ JWT authentication successful, proceeding to endpoint")
        except Exception as e:
            logger.error("❌ JWT authentication failed: %s", str(e))
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

@app.route('/scryforge/api/v1/image/arucolocations', methods=['POST'])
@jwt_required
@limiter.limit("20 per second")
def get_aruco_positions():
    """POST /image/arucolocations - Get ArUco marker positions from image"""
    logger.info("🔄 Processing ArUco detection request")
    
    try:
        if 'image' not in request.files:
            logger.error("❌ No image file provided in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        logger.info("📁 Received image file: %s (content-type: %s)", file.filename, file.content_type)
        
        if not file.content_type.startswith('image/'):
            logger.error("❌ Invalid file type: %s", file.content_type)
            return jsonify({'error': 'File must be an image'}), 400

        # Read image bytes and convert to numpy array
        logger.info("🔄 Reading and decoding image...")
        try:
            image_bytes = file.read()
            logger.info("📊 Image size: %d bytes", len(image_bytes))
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("❌ Failed to decode image - cv2.imdecode returned None")
                raise ValueError("Could not decode image")
                
            logger.info("✅ Image decoded successfully - shape: %s", str(image.shape))
            
        except Exception as e:
            logger.error("❌ Image decoding failed: %s", str(e))
            return jsonify({'error': f'Could not decode image: {str(e)}'}), 400
        
        # Get marker positions
        logger.info("🔍 Starting ArUco marker detection...")
        try:
            positions = aruco_detector.get_central_coordinates(image)
            logger.info("✅ ArUco detection completed - positions: %s", str(positions))
            
        except Exception as e:
            logger.error("❌ ArUco detection failed: %s", str(e))
            return jsonify({'error': f'ArUco detection failed: {str(e)}'}), 500
        
        if positions is None:
            logger.warning("⚠️ Could not detect all required markers")
            return jsonify({'error': 'Could not detect all markers'}), 400
            
        logger.info("✅ Successfully returning ArUco positions")
        return jsonify({'positions': positions})
        
    except Exception as e:
        logger.error("❌ Unexpected error in get_aruco_positions: %s", str(e))
        logger.exception("Full traceback:")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/scryforge/api/v1/image/categories/positions', methods=['POST'])
@jwt_required
@limiter.limit("2 per second")
def process_category_positions():
    """POST /image/categories/positions - Detect categories in uploaded image"""
    logger.info("🔄 Processing category detection request")
    
    try:
        if 'image' not in request.files:
            logger.error("❌ No image file provided in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        if not models_ready:
            logger.error("❌ Models not ready - still loading")
            return jsonify({"error": "Model still loading. Try again shortly."}), 503
            
        file = request.files['image']
        logger.info("📁 Received image file: %s (content-type: %s)", file.filename, file.content_type)
        
        if not file.content_type.startswith('image/'):
            logger.error("❌ Invalid file type: %s", file.content_type)
            return jsonify({'error': 'File must be an image'}), 400

        # Read image bytes and convert to numpy array
        logger.info("🔄 Reading and decoding image...")
        try:
            image_bytes = file.read()
            logger.info("📊 Image size: %d bytes", len(image_bytes))
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("❌ Failed to decode image - cv2.imdecode returned None")
                raise ValueError("Could not decode image")
                
            logger.info("✅ Image decoded successfully - shape: %s", str(image.shape))
            
        except Exception as e:
            logger.error("❌ Image decoding failed: %s", str(e))
            return jsonify({'error': f'Could not decode image: {str(e)}'}), 400
        
        # Detect categories
        logger.info("🔍 Starting category detection...")
        try:
            detected_categories = category_detector.detect_bases(image)
            logger.info("✅ Category detection completed - found %d categories", len(detected_categories))
            
        except Exception as e:
            logger.error("❌ Category detection failed: %s", str(e))
            return jsonify({'error': f'Category detection failed: {str(e)}'}), 500
        
        # Convert to pixel coordinates
        logger.info("🔄 Converting to pixel coordinates...")
        try:
            height, width = image.shape[:2]
            pixel_positions = [{
                'category': pos.category.value,
                'x': int(pos.location[0] * width),
                'y': int(pos.location[1] * height),
                'width': int(pos.bounding_rect[2] * width),
                'height': int(pos.bounding_rect[3] * height)
            } for pos in detected_categories]
            
            logger.info("✅ Successfully converted %d positions to pixel coordinates", len(pixel_positions))
            
        except Exception as e:
            logger.error("❌ Failed to convert to pixel coordinates: %s", str(e))
            return jsonify({'error': f'Failed to convert coordinates: {str(e)}'}), 500
        
        logger.info("✅ Successfully returning category positions")
        return jsonify({'positions': pixel_positions})
        
    except Exception as e:
        logger.error("❌ Unexpected error in process_category_positions: %s", str(e))
        logger.exception("Full traceback:")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
