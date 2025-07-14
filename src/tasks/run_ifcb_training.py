from prefect import task
import docker
import os

from prefect import get_run_logger

from src.prov import on_task_complete
from src.params.params_ifcb_flow_metric import IFCBTrainingParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_ifcb_training(ifcb_training_params: IFCBTrainingParams, ifcb_image: str):
    """
    Run IFCB flow metric model training in a Docker container.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        ifcb_training_params.data_dir: {'bind': '/app/data', 'mode': 'ro'},
        ifcb_training_params.output_dir: {'bind': '/app/output', 'mode': 'rw'}
    }
    
    # Mount id_file if provided
    id_file_container_path = None
    if ifcb_training_params.id_file is not None:
        id_file_container_path = '/app/ids.txt'
        volumes[ifcb_training_params.id_file] = {'bind': id_file_container_path, 'mode': 'ro'}
    
    # Build command arguments
    command_args = [
        "python", "train.py",
        "/app/data",
        "--n-jobs", str(ifcb_training_params.n_jobs),
        "--contamination", str(ifcb_training_params.contamination),
        "--aspect-ratio", str(ifcb_training_params.aspect_ratio),
        "--chunk-size", str(ifcb_training_params.chunk_size),
        "--model", f"/app/output/{ifcb_training_params.model_filename}"
    ]
    
    # Add optional id-file flag if provided
    if ifcb_training_params.id_file is not None:
        command_args.extend(["--id-file", id_file_container_path])
    
    logger.info(f'Running container with command: {" ".join(command_args)}')
    
    try:
        container = client.containers.run(
            ifcb_image,
            command_args,
            volumes=volumes,
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