from pydantic import BaseModel


class SegGPTRequest(BaseModel):
    input_dir: str
    prompt_dir: str
    target_dir: str
    output_dir: str
    patch_images: bool
    num_prompts: int = 0


class InfrastructureParams(BaseModel):
    tmp_dir: str = "/tmp/prefect"
