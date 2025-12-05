from pydantic import BaseModel, Field
from typing import Optional, Literal

# Parameters relevant to ONNX inference
class ONNXInferenceParams(BaseModel):
    model: str = Field(..., description="Path to the ONNX model file")
    input_dir: str = Field(..., description="Directory containing input data")
    output_dir: str = Field(..., description="Directory where results will be saved")
    batch: Optional[int] = Field(None, description="Batch size for inference")
    classes: Optional[str] = Field(None, description="Class file (txt) containing the classes to use in the results csv")
    outfile: Optional[str] = Field(None, description="Custom output filename")
    subfolder_type: Literal["run-date", "model-name"] = Field("model-name", description="Toggle between using run date or model name (default) for output directory structure")
    force_notorch: Optional[bool] = Field(None, description="Force non-PyTorch backend")
    cuda_visible_devices: str = Field("0,1,2,3", description="GPU devices to use")
    ensure_softmax: Optional[bool] = Field(None, description="Ensure softmax is applied to model output")
