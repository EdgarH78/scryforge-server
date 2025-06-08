# Base image with Python + Torch
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl unzip wget rclone \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app ./app
COPY entrypoint.sh .

# Ensure script is executable
RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 8000

# Run the entrypoint
CMD ["./entrypoint.sh"]
