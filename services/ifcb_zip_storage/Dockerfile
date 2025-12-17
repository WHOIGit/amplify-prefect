# Use slim Python base image
FROM python:3.13-slim

# Set environment variables to prevent .pyc files and set UTF-8 encoding
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Copy application files
COPY services/ifcb_zip_storage/requirements.txt .
COPY services/ifcb_zip_storage/src/ /app/src/

# Install uv and dependencies
RUN pip install uv && \
    uv pip install --system -r requirements.txt

# Default command (allows overriding with arguments)
ENTRYPOINT ["python"]
