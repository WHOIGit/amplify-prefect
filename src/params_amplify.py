from pydantic import BaseModel


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
