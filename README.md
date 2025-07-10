# AMPLIfy Prefect Pipeline

A Prefect server for orchestrating machine learning training and inference runs. Supports YOLO (training and inference), SegGPT (inference), and ONNX models (inference). After setting up the system with Docker, users can monitor and run the workflows using the UI accessible in a browser.

## Setup

### 1. Environment Configuration

Copy the example environment file and fill in your values:
```bash
cp .env.example .env
```

Edit `.env` with your specific values:
- `POSTGRES_USERNAME`: Your PostgreSQL username
- `POSTGRES_PASSWORD`: Your PostgreSQL password  
- `EXTERNAL_HOST_NAME`: External hostname of your machine
- `PROVENANCE_STORE_URL`: URL for provenance store
- `MEDIASTORE_URL`: URL for your media store
- `MEDIASTORE_TOKEN`: Authentication token for media store

### 2. Launch PostgreSQL Database

Use Docker Compose to start the PostgreSQL container:
```bash
docker compose up -d postgres
```

### 3. Python Environment Setup

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

### 4. Configure and Start Prefect Server

Load environment variables and configure Prefect:
```bash
# Load environment variables
source .env

# Set Prefect configuration
prefect config set PREFECT_SERVER_API_HOST="$EXTERNAL_HOST_NAME"
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://$POSTGRES_USERNAME:$POSTGRES_PASSWORD@localhost:5432/prefect"

# Start Prefect server
prefect server start
```

### 5. Deploy Workflows

In separate terminal windows, deploy the workflows you want to use:

**For ONNX Inference:**
```bash
source .venv/bin/activate
source .env
python src/flows/onnx_inference.py
```

**For YOLO Inference:**
```bash
source .venv/bin/activate  
source .env
python src/flows/yolo_inference.py
```

## Using the Workflows

Navigate to the Prefect UI in your browser at `http://{EXTERNAL_HOST_NAME}:4200`

### ONNX Inference Workflow

The ONNX inference workflow requires the following parameters in the Prefect UI:

**ONNXInferenceParams:**
- `model_dir`: Directory containing the ONNX model files
- `input_dir`: Directory containing input data
- `output_dir`: Directory where results will be saved
- `model_name`: Name of the ONNX model file
- `path_to_bin_dir`: Path to binary directory within input_dir
- `batch` (optional): Batch size for inference
- `classes` (optional): Specific classes to process
- `outfile` (optional): Custom output filename
- `force_notorch` (optional): Force non-PyTorch backend
- `cuda_visible_devices`: GPU devices to use (default: "0,1,2,3")

### YOLO Inference Workflow

The YOLO inference workflow requires two parameter sets:

**YOLOInferenceParams:**
- `data_dir`: Directory containing input images/videos
- `output_dir`: Directory where results will be saved
- `model_weights_path`: Path to YOLO model weights (.pt file)
- `device`: Compute device for inference (e.g., "0" for GPU 0, "cpu")
- `agnostic_nms`: Class-agnostic Non-Maximum Suppression (default: true)
- `iou`: IoU threshold for NMS to eliminate overlapping boxes (default: 0.5)
- `conf`: Minimum confidence threshold for detections (default: 0.1)
- `imgsz`: Image size for inference (default: 1280)
- `batch`: Batch size for processing multiple inputs (default: 16)
- `half`: Half-precision (FP16) inference for speed (default: false)
- `max_det`: Maximum detections allowed per image (default: 300)
- `vid_stride`: Frame stride for video processing (default: 1)
- `stream_buffer`: Queue frames vs drop old frames (default: false)
- `visualize`: Visualize model features during inference (default: false)
- `augment`: Test-time augmentation for improved robustness (default: false)
- `classes`: Filter predictions to specific class IDs (optional)
- `retina_masks`: High-resolution segmentation masks (default: false)
- `embed`: Extract feature vectors from specified layers (optional)
- `name`: Name for prediction run subdirectory (optional)
- `verbose`: Display detailed inference logs (default: true)

**YOLOVisualizationParams:**
- `show`: Display annotated images/videos in window (default: false)
- `save`: Save annotated images/videos to file (default: false)
- `save_frames`: Save individual video frames as images (default: false)
- `save_txt`: Save detection results in text format (default: false)
- `save_conf`: Include confidence scores in saved text files (default: false)
- `save_crop`: Save cropped images of detections (default: false)
- `show_labels`: Display labels for each detection (default: true)
- `show_conf`: Display confidence scores alongside labels (default: true)
- `show_boxes`: Draw bounding boxes around detected objects (default: true)

## YOLO Training Data Format

For YOLO training, the data directory should contain a `dataset.yaml` file:

```yaml
path: /data # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path') 
test: images/test # test images (optional)

names:
  0: person
  1: bicycle
  2: car
```

Ensure your directory structure matches:
```
data_dir/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/ (optional)
└── labels/
    ├── train/
    ├── val/
    └── test/ (optional)
``` 
