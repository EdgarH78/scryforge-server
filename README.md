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

- Real-time camera feed processing
- Multiple base color detection (Red, Blue, Green, Violet, Yellow, Orange, etc.)
- Camera calibration using ArUco markers
- Adjustable camera settings (rotation, flip, scale)
- Training data capture capabilities
- Web-based user interface
- Support for multiple cameras

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Webcam or USB camera

## Installation

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

4. Download required model files:
- Place `fasterrcnn_token_detector.pth` in the project root directory
- Place `sam_vit_h_4b8939.pth` in the project root directory if using SAM features

## Project Structure

```
scryforge/
├── scryforge/           # Main package directory
│   ├── detector.py     # Base detection implementation
│   ├── camera.py       # Camera handling
│   ├── server.py       # Flask server implementation
│   ├── session.py      # Session management
│   └── lens.py         # Image processing pipeline
├── training/           # Training data directory
├── dataset/           # Dataset for model training
├── requirements.txt    # Project dependencies
├── setup.py           # Package setup file
└── main.py            # Application entry point
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
- Select and configure cameras
- Toggle detection
- Adjust camera settings
- Capture training data
- View detection results

## Camera Calibration

1. Print the ArUco markers (IDs 0-3)
2. Place markers in clockwise order:
   - ID 0: Top-left
   - ID 1: Top-right
   - ID 2: Bottom-right
   - ID 3: Bottom-left
3. Use the web interface to perform calibration

## Training Data Collection

The system can capture training data for improving detection accuracy:

1. Enable training data capture in the web interface
2. Position bases in the camera view
3. Training images and labels are saved to:
   - Images: `training/images/`
   - Labels: `training/labels/`

## Development

### Running Tests
```bash
pytest
```

### Code Structure

- `detector.py`: Implements base detection using CNN models
- `camera.py`: Handles camera initialization and frame capture
- `server.py`: Implements the Flask web server and API endpoints
- `session.py`: Manages application state and configuration
- `lens.py`: Implements image processing pipeline

## License

This codebase is released under a dual license:

1. **Open Source License** - MIT License
   - You can use, modify, and distribute the code
   - You can train your own models
   - You must include the original copyright notice

2. **Commercial License**
   - Required for accessing our pre-trained models
   - Includes commercial support
   - Access to our hosted service
   - Contact [your-contact] for pricing

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