from pydantic import BaseModel
from typing import Optional

# Parameters relevant to ONNX inference
class ONNXInferenceParams(BaseModel):
    model_dir: str
    input_dir: str
    output_dir: str
    model_name: str
    batch: Optional[int] = None
    classes: Optional[str] = None
    outfile: Optional[str] = None
    force_notorch: Optional[bool] = None
    cuda_visible_devices: str = "0,1,2,3"
