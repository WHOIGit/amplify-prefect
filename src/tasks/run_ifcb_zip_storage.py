from prefect import task, get_run_logger
import docker
from dotenv import dotenv_values

from src.params.params_ifcb_zip_storage import IFCBZipStorageParams


@task(log_prints=True)
def run_ifcb_zip_storage(params: IFCBZipStorageParams, image: str):
    """
    Run IFCB ZIP generation and storage in a Docker container.

    The container will:
    1. Use pyifcb to iterate through IFCB data
    2. Generate ZIP streams for each fileset
    3. Upload to object store defined by storage YAML
    """
    client = docker.from_env()
    logger = get_run_logger()

    # Set up volumes
    volumes = {
        params.data_dir: {'bind': '/data/ifcb', 'mode': 'ro'},
        params.storage_yaml: {'bind': '/config/storage.yaml', 'mode': 'ro'}
    }

    # Load environment variables from env file
    environment = {}
    if params.env_file:
        logger.info(f"Loading environment variables from: {params.env_file}")
        environment = dict(dotenv_values(params.env_file))
        logger.info(f"Loaded {len(environment)} environment variables")

    # Build command arguments
    command_args = [
        "/app/src/process_ifcb_zips.py",
        "--data-dir", "/data/ifcb",
        "--storage-config", "/config/storage.yaml",
        "--num-workers", str(params.num_workers)
    ]

    logger.info(f'Running IFCB ZIP storage with command: {" ".join(command_args)}')

    try:
        # Run container in detached mode to stream logs properly
        container = client.containers.run(
            image,
            command_args,
            volumes=volumes,
            environment=environment,
            detach=True
        )

        # Stream logs in real-time
        try:
            for log_line in container.logs(stream=True, follow=True):
                logger.info(log_line.decode('utf-8').rstrip())

            # Wait for container to finish
            result = container.wait()
            exit_code = result['StatusCode']

            if exit_code != 0:
                raise RuntimeError(f"Docker container failed with exit code {exit_code}")

        finally:
            # Clean up container
            try:
                container.remove()
            except:
                pass

        logger.info("IFCB ZIP storage completed successfully")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
