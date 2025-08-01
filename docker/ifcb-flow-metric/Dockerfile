# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

# Clone the ifcb-flow-metric repository
RUN git clone -b feature-isolation https://github.com/WHOIGit/ifcb-flow-metric.git /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Default command runs train.py
CMD ["python", "train.py"]
