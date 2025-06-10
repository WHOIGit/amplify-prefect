from enum import Enum

from pydantic import BaseModel


class YoloModeEnum(str, Enum):
    train = 'train'
    inference = 'inference'


# Parameters relevant to SegGPT inference
class SegGPTRequest(BaseModel):
    input_dir: str
    prompt_dir: str
    target_dir: str
    output_dir: str
    patch_images: bool
    num_prompts: int = 0


# Parameters relevant to container infrastructure
class InfrastructureParams(BaseModel):
    tmp_dir: str = "/tmp/prefect"


class YOLOTrainParams(BaseModel):
    data_dir: str
    output_dir: str
    model_name: str
    epochs: int
    gpus: str


class YOLOInferenceParams(BaseModel):
    data_dir: str
    output_dir: str
    model_weights_path: str
    device: str
    agnostic_nms: bool = True
    iou: float = 0.5
    conf: float = 0.1
    imgsz: int = 1280
    batch: int = 16
    half: bool = False
    max_det: int = 300
    vid_stride: int = 1
    stream_buffer: bool = False
    visualize: bool = False
    augment: bool = False
    classes: list[int] | None = None
    retina_masks: bool = False
    embed: list[int] | None = None
    name: str | None = None
    verbose: bool = True


class YOLOVisualizationParams(BaseModel):
    show: bool = False
    save: bool = False
    save_frames: bool = False
    save_txt: bool = False
    save_conf: bool = False
    save_crop: bool = False
    show_labels: bool = True
    show_conf: bool = True
    show_boxes: bool = True