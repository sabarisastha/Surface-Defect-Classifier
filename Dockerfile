# Dockerfile for the Surface Defect Classifier Flask app
# Build: docker build -t defect-classifier .
# Run:   docker run -p 5000:5000 -v $(pwd)/models:/app/models defect-classifier

FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Environment
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app/app.py"]
