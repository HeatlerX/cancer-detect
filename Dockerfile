# Cloud Run builds this remotely via `gcloud run deploy --source .`
# You do NOT need Docker installed locally to use this file.

FROM python:3.11-slim

WORKDIR /app

# Required by opencv-python (a dependency of ultralytics) to run headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: smaller download, faster build, plenty fast enough for
# a small YOLO11n model on Cloud Run's CPU instances
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT automatically at runtime (usually 8080) --
# app.py already reads it, so nothing to hardcode here.
CMD ["python", "app.py"]
