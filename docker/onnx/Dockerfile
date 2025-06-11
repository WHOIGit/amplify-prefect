# Use slim Python base image
FROM python:3.12-slim

# Set environment variables to prevent .pyc files and set UTF-8 encoding
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install git and other required tools, clone the repo, install dependencies, and clean up
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/WHOIGit/amplify_onnx_inference.git /app && \
    pip install -r /app/requirements.txt && \
    apt-get remove -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Default command (allows overriding with arguments)
ENTRYPOINT ["python", "src/infer_ifcbbins.py"]
