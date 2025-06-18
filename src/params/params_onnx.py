from pydantic import BaseModel

# Parameters relevant to ONNX inference
class ONNXInferenceParams(BaseModel):
    model_dir: str
    input_dir: str
    output_dir: str
    model_name: str
    path_to_bin_dir: str
    batch: int
    classes: str
    outdir: str
    outfile: str
    force-notorch: bool
