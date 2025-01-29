# Stage 1: Build with dependencies
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /workspace/src/

# Copy requirements file
COPY src/requirements.txt .

# Install build tools and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image without build tools
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace/src/

# Copy installed libraries from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Update PATH environment variable
ENV PATH="/usr/local/bin:${PATH}"

# Copy application code
COPY src/ /workspace/src/

# Expose port and set command
EXPOSE 4200
CMD ["/bin/bash", "./start.sh"]