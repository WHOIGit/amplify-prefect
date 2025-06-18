from prefect import task
import subprocess

from prefect import get_run_logger

from src.prov import on_task_complete
from src.params.params_onnx import ONNXInferenceParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_onnx_inference(onnx_inference_params: ONNXInferenceParams, onnx_image: str):
    """
    Run inference with an ONNX model in a Podman container.
    """

    command = (
        f"podman run "
        f"-it --rm --gpus all --ipc host "
        f"-e CUDA_VISIBLE_DEVICES=1"
        f"-v {onnx_inference_params.model_dir}/models:/app/models "
        f"-v {onnx_inference_params.input_dir}/inputs:/app/inputs "
        f"-v {onnx_inference_params.output_dir}/outputs:/app/outputs "
        f"{onnx_image} "
        f"models/{onnx_inference_params.model_name} inputs/{onnx_inference_params.path_to_bin_dir}"
        f"--batch {onnx_inference_params.batch} "
        f"--classes {onnx_inference_params.classes} "
        f"--outdir {onnx_inference_params.outdir} "
        f"--outfile {onnx_inference_params.outfile} "
        f"--force-notorch {onnx_inference_params.force_notorch} "
        )
    
    logger = get_run_logger()
    logger.info(f'command: {command}')

    process = subprocess.Popen(
        command, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    for line in process.stdout:
        logger.info(line)

