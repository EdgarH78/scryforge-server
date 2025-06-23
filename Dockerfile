FROM python:3.10-slim

# Install git and system deps before pip install
RUN apt-get update && apt-get install -y \
    git curl unzip wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /scryforge

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY app/models/fasterrcnn_token_detector_scripted.pt ./app/models/

CMD gunicorn --workers=1 --bind 0.0.0.0:$PORT app.server:app