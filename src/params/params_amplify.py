from enum import Enum

from pydantic import BaseModel, Field


class YoloModeEnum(str, Enum):
    train = 'train'
    inference = 'inference'


# Parameters relevant to SegGPT inference
class SegGPTRequest(BaseModel):
    input_dir: str = Field(..., description="Directory containing input images")
    prompt_dir: str = Field(..., description="Directory containing prompt images")
    target_dir: str = Field(..., description="Directory containing target images")
    output_dir: str = Field(..., description="Directory where results will be saved")
    patch_images: bool = Field(..., description="Whether to patch images during processing")
    num_prompts: int = Field(0, description="Number of prompts to use")


# Parameters relevant to container infrastructure
class InfrastructureParams(BaseModel):
    tmp_dir: str = Field("/tmp/prefect", description="Temporary directory for Prefect operations")


class YOLOTrainParams(BaseModel):
    data_dir: str = Field(..., description="Directory containing training dataset with dataset.yaml")
    output_dir: str = Field(..., description="Directory where training results will be saved")
    model_name: str = Field(..., description="Name of the YOLO model to train")
    epochs: int = Field(..., description="Number of training epochs")
    gpus: str = Field(..., description="GPU devices to use for training")


class YOLOInferenceParams(BaseModel):
    data_dir: str = Field(..., description="Directory containing input images/videos")
    output_dir: str = Field(..., description="Directory where results will be saved")
    model_weights_path: str = Field(..., description="Path to YOLO model weights (.pt file)")
    device: str = Field(..., description="Compute device for inference (e.g., '0' for GPU 0, 'cpu')")
    agnostic_nms: bool = Field(True, description="Class-agnostic Non-Maximum Suppression")
    iou: float = Field(0.5, description="IoU threshold for NMS to eliminate overlapping boxes")
    conf: float = Field(0.1, description="Minimum confidence threshold for detections")
    imgsz: int = Field(1280, description="Image size for inference")
    batch: int = Field(16, description="Batch size for processing multiple inputs")
    half: bool = Field(False, description="Half-precision (FP16) inference for speed")
    max_det: int = Field(300, description="Maximum detections allowed per image")
    vid_stride: int = Field(1, description="Frame stride for video processing")
    stream_buffer: bool = Field(False, description="Queue frames vs drop old frames")
    visualize: bool = Field(False, description="Visualize model features during inference")
    augment: bool = Field(False, description="Test-time augmentation for improved robustness")
    classes: list[int] | None = Field(None, description="Filter predictions to specific class IDs")
    retina_masks: bool = Field(False, description="High-resolution segmentation masks")
    embed: list[int] | None = Field(None, description="Extract feature vectors from specified layers")
    name: str | None = Field(None, description="Name for prediction run subdirectory")
    verbose: bool = Field(True, description="Display detailed inference logs")


class YOLOVisualizationParams(BaseModel):
    show: bool = Field(False, description="Display annotated images/videos in window")
    save: bool = Field(False, description="Save annotated images/videos to file")
    save_frames: bool = Field(False, description="Save individual video frames as images")
    save_txt: bool = Field(False, description="Save detection results in text format")
    save_conf: bool = Field(False, description="Include confidence scores in saved text files")
    save_crop: bool = Field(False, description="Save cropped images of detections")
    show_labels: bool = Field(True, description="Display labels for each detection")
    show_conf: bool = Field(True, description="Display confidence scores alongside labels")
    show_boxes: bool = Field(True, description="Draw bounding boxes around detected objects")