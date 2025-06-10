import os
from pathlib import Path
import shutil

from prefect import flow

from params_amplify import SegGPTRequest, InfrastructureParams
from tasks.run_seggpt import request
from tasks.upload_media import upload
from tasks.download_media import download


@flow(log_prints=True)
def run_seggpt(request_params: SegGPTRequest, infra_params: InfrastructureParams):
    """Flow: Run SegGPT inference using the given parameters."""

    os.makedirs(infra_params.tmp_dir, exist_ok=True)

    for img_path in Path(request_params.input_dir).iterdir():
        img_key = img_path.name
        upload(img_path.resolve(), img_key)
        download(img_key, infra_params.tmp_dir)

    request(
        infra_params.tmp_dir,
        request_params.prompt_dir,
        request_params.target_dir,
        request_params.output_dir,
        request_params.patch_images,
        request_params.num_prompts,
    )

    shutil.rmtree(infra_params.tmp_dir)


# Deploy the flow
if __name__ == "__main__":
    run_seggpt.serve(name="seggpt-inference")
