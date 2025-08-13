from prefect import task
import docker
import os

from prefect import get_run_logger

from src.prov import on_task_complete
from src.params.params_extract_slim_features import ExtractSlimFeaturesParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_extract_slim_features(extract_features_params: ExtractSlimFeaturesParams, extract_features_image: str):
    """
    Run extract_slim_features.py in a Docker container to extract IFCB features.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        extract_features_params.data_directory: {'bind': '/app/data', 'mode': 'ro'},
        extract_features_params.output_directory: {'bind': '/app/output', 'mode': 'rw'}
    }
    
    # Build command arguments (ENTRYPOINT already includes "python extract_slim_features.py")
    command_args = [
        "/app/data",
        "/app/output"
    ]
    
    # Add optional bins flag if provided
    if extract_features_params.bins is not None and len(extract_features_params.bins) > 0:
        command_args.extend(["--bins"] + extract_features_params.bins)
    
    logger.info(f'Running container with command: {" ".join(command_args)}')
    
    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()
        
        container = client.containers.run(
            extract_features_image,
            command_args,
            volumes=volumes,
            user=f"{uid}:{gid}",
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
