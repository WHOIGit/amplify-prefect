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

    command_parts = [
        "podman run",
        "-it --rm --gpus all --ipc host",
        "-e CUDA_VISIBLE_DEVICES=1",
        f"-v {onnx_inference_params.model_dir}/models:/app/models",
        f"-v {onnx_inference_params.input_dir}/inputs:/app/inputs",
        f"-v {onnx_inference_params.output_dir}/outputs:/app/outputs",
        onnx_image,
        f"models/{onnx_inference_params.model_name}",
        f"inputs/{onnx_inference_params.path_to_bin_dir}"
    ]
    
    # Add optional flags only if they are specified
    if onnx_inference_params.batch is not None:
        command_parts.append(f"--batch {onnx_inference_params.batch}")
    if onnx_inference_params.classes is not None:
        command_parts.append(f"--classes {onnx_inference_params.classes}")
    if onnx_inference_params.outdir is not None:
        command_parts.append(f"--outdir {onnx_inference_params.outdir}")
    if onnx_inference_params.outfile is not None:
        command_parts.append(f"--outfile {onnx_inference_params.outfile}")
    if onnx_inference_params.force_notorch is not None and onnx_inference_params.force_notorch:
        command_parts.append("--force_notorch")
    
    command = " ".join(command_parts)
    
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

