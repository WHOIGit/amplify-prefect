import os
from typing import Literal, List

from prefect import task, get_run_logger
import docker
from pydantic import BaseModel, Field

from dojo.schemas import TrainingRunConfig
from src.tasks.gpu import allocate_gpus, run_container

DOJO_IMAGE_ENV = "AMPLIFY_DOJO_IMAGE"

class VolumeMapping(BaseModel):
    host_path: str = Field(..., description="Path on the host machine")
    container_path: str = Field(..., description="Path inside the container")
    mode: Literal['ro','rw']= Field('rw', description="Mount mode: read-only ('ro') or read-write ('rw')")

@task()
def run_dojo_container(output_dir: str, input_volumes: List[VolumeMapping], subcommands:List[str], training_run_config: TrainingRunConfig, device_ids:List[str]=['all']):
    """
    Run Image Classifier Dojo in a Docker container.

    ``device_ids`` is a GPU spec (see src/tasks/gpu.py); ['all'] means every idle GPU.
    """

    image = os.environ.get(DOJO_IMAGE_ENV)
    if not image:
        raise RuntimeError(
            f"{DOJO_IMAGE_ENV} is not set. Set it to the Image Classifier Dojo image "
            "(<registry>/amplify/image_classifier_dojo:<tag>) in the serve process's environment, "
            "e.g. in ~/.config/amplify-prefect/flows.env."
        )
    client = docker.from_env()
    logger = get_run_logger()

    volumes = {
        output_dir: {'bind': '/workspace/experiments', 'mode': 'rw'}
    }
    for vol in input_volumes:
        volumes[vol.host_path] = {'bind': vol.container_path, 'mode': vol.mode}

    command = f"{' '.join(subcommands)}' " \
              f"--logger {training_run_config.logger.model_dump_json()} " \
              f"--dataset_config {training_run_config.dataset_config.model_dump_json()} " \
              f"--model {training_run_config.model.model_dump_json()} " \
              f"--training {training_run_config.training.model_dump_json()} " \
              f"--runtime {training_run_config.runtime.model_dump_json()} "
    logger.info(command)
    command = subcommands + [
                "--logger", training_run_config.logger.model_dump_json(),
                "--dataset_config", training_run_config.dataset_config.model_dump_json(),
                "--model", training_run_config.model.model_dump_json(),
                "--training", training_run_config.training.model_dump_json(),
                "--runtime", training_run_config.runtime.model_dump_json()
            ]
    logger.info(command)
    with allocate_gpus(device_ids, logger) as gpus:
        run_container(
            client,
            image,
            command,
            logger,
            gpus=gpus,
            shm_size='8g',
            volumes=volumes,
            ipc_mode="host",
        )
    logger.info("✓ Training completed successfully")
