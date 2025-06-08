# ScryForge

ScryForge is a commercial computer vision system designed to detect and classify colored bases in real-time camera feeds. It uses advanced computer vision techniques including ArUco marker detection for calibration and a trained CNN model for base detection.

## Repository Status

This is a read-only repository maintained by the ScryForge team. While you can fork the repository and modify it for your own use under the terms of the license, we do not accept external contributions. This allows us to maintain strict quality control and ensure the codebase meets our standards.

## Commercial Usage

This repository contains the core codebase for ScryForge. While the code is open source, the trained models and training data are not included. To use the fully trained system:

1. **Commercial Service**: Subscribe to our hosted service at [your-service-url]
2. **Self-Hosted Option**: You can train your own models using your own dataset following our documentation

The repository includes:
- ✅ Complete source code
- ✅ Documentation
- ✅ Training pipeline
- ❌ Trained models
- ❌ Training dataset
- ❌ Commercial support

## Features

- Real-time base detection API
- Multiple base color detection (Red, Blue, Green, Violet, Yellow, Orange, etc.)
- Camera calibration using ArUco markers
- Docker containerization
- RESTful API endpoints
- Support for image upload and processing

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Docker (optional, for containerized deployment)

## Project Structure

```
scryforge/
├── app/                # Main application directory
│   ├── server.py      # Flask server and API endpoints
│   ├── detector.py    # Detection implementations
│   └── main.py        # Application entry point
├── training/          # Training utilities
├── dataset/          # Dataset for model training (not included)
├── Dockerfile        # Docker configuration
└── requirements.txt  # Python dependencies
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone [your-repo-url]
cd scryforge
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Deployment

1. Build the container:
```bash
docker build -t scryforge .
```

2. Run the container:
```bash
docker run -p 5000:5000 scryforge
```

## API Documentation

The complete API specification is available in OpenAPI (Swagger) format in `swagger.yml`. You can view this documentation using any OpenAPI viewer or import it into tools like Postman.

## API Endpoints

### ArUco Marker Detection
```http
POST /image/arucolocations
Content-Type: multipart/form-data

file: <image_file>
```

### Base Detection
```http
POST /image/categories/positions
Content-Type: multipart/form-data

file: <image_file>
```

## Training Your Own Models

While we don't provide our training data or pre-trained models, you can train your own:

1. Create a dataset following our format:
   ```
   dataset/
   ├── images/          # Your base images
   └── labels/          # Corresponding labels
   ```

2. Use our training scripts:
   ```bash
   python augment_dataset.py  # Data augmentation
   # Additional training steps documented in our wiki
   ```

3. Place your trained models in the project root:
   - `fasterrcnn_token_detector.pth`
   - `sam_vit_h_4b8939.pth` (if using SAM features)

## Support

For access to our pre-trained models and hosted service:
- Email: [your-email]
- Website: [your-website]
- Documentation: [your-docs-url]

## Disclaimer

This repository does not include trained models or training data. These are available separately through our commercial offering. The codebase is maintained exclusively by the ScryForge team to ensure consistency and quality. 