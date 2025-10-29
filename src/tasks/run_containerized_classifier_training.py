from prefect import task, get_run_logger
import docker

from prov import on_task_complete
from image_classifier_dojo.schemas import TrainingRunConfig

@task(on_completion=[on_task_complete])
def run_container(data_dir: str, output_dir: str, subcommands:list, training_run_config: TrainingRunConfig):
    """
    Run Image Classifier Dojo in a Docker container.
    """

    client = docker.from_env()
    logger = get_run_logger()

    volumes = {
        data_dir: {'bind': '/workspace/datasets', 'mode': 'rw'},
        output_dir: {'bind': '/app/experiments', 'mode': 'rw'}
    }

    command = f"{' '.join(subcommands)}' " \
              f"--logger {training_run_config.logger.model_dump_json()} " \
              f"--dataset_config {training_run_config.dataset_config.model_dump_json()} " \
              f"--model {training_run_config.model.model_dump_json()} " \
              f"--training {training_run_config.training.model_dump_json()} " \
              f"--runtime {training_run_config.runtime.model_dump_json()} "
    try:
        container = client.containers.run(
            'harbor-registry.whoi.edu/amplify/image_classifier_dojo:v0.2.0',
            command,
            shm_size='8g',
            volumes=volumes,
            device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
            ipc_mode="host",
            remove=False,  # Don't auto-remove so we can get logs on failure
            detach=True,
        )

    # Stream logs in real-time
        try:
            for line in container.logs(stream=True, follow=True):
                logger.info(line.decode('utf-8').rstrip())
        except Exception as log_error:
            logger.error(f"Error streaming logs: {log_error}")

        # Wait for container to finish and check exit code
        result = container.wait()
        exit_code = result['StatusCode']

        if exit_code != 0:
            # Get the full logs for error reporting
            full_logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            container.remove()
            logger.error(f"Container failed with exit code {exit_code}")
            logger.error(f"Full container output:\n{full_logs}")
            raise RuntimeError(f"Docker container failed with exit code {exit_code}")

        # Clean up container on success
        container.remove()
        logger.info("✓ Validation completed successfully")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with exit code {e.exit_status}")
        logger.error(f"Command: {e.command}")
        if e.stderr:
            stderr = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else str(e.stderr)
            logger.error(f"Container stderr:\n{stderr}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise