from prefect import task
import docker

from prefect import get_run_logger

from src.prov import on_task_complete
from src.params.params_onnx import ONNXInferenceParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_onnx_inference(onnx_inference_params: ONNXInferenceParams, onnx_image: str):
    """
    Run inference with an ONNX model in a Docker container.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        onnx_inference_params.model_dir: {'bind': '/app/models', 'mode': 'rw'},
        onnx_inference_params.input_dir: {'bind': '/app/inputs', 'mode': 'rw'},
        onnx_inference_params.output_dir: {'bind': '/app/outputs', 'mode': 'rw'}
    }
    
    # Mount classes file if provided
    classes_container_path = None
    if onnx_inference_params.classes is not None:
        classes_container_path = '/app/classes.txt'
        volumes[onnx_inference_params.classes] = {'bind': classes_container_path, 'mode': 'ro'}
    
    # Set up environment variables
    environment = {
        'CUDA_VISIBLE_DEVICES': onnx_inference_params.cuda_visible_devices
    }
    
    # Build command arguments
    command_args = [
        f"models/{onnx_inference_params.model_name}",
        "inputs"
    ]
    
    # Add optional flags only if they are specified
    if onnx_inference_params.batch is not None:
        command_args.extend(["--batch", str(onnx_inference_params.batch)])
    if onnx_inference_params.classes is not None:
        command_args.extend(["--classes", classes_container_path])
    if onnx_inference_params.outfile is not None:
        command_args.extend(["--outfile", onnx_inference_params.outfile])
    if onnx_inference_params.force_notorch is not None and onnx_inference_params.force_notorch:
        command_args.append("--force_notorch")
    
    logger.info(f'Running container with command: {" ".join(command_args)}')
    
    try:
        container = client.containers.run(
            onnx_image,
            command_args,
            volumes=volumes,
            environment=environment,
            device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
            ipc_mode="host",
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
            stream=True
        )
        
        # Stream output
        for line in container:
            logger.info(line.decode('utf-8').rstrip())
            
    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise

