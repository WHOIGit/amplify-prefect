import os
from typing import Literal, List, Optional, Union

from prefect import task, get_run_logger
import docker
from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from dojo.schemas.core import AimLoggerConfig, S3Config, TrainingRunConfig
from src.tasks.gpu import allocate_gpus, run_container

DOJO_IMAGE_ENV = "AMPLIFY_DOJO_IMAGE"
ARTIFACTS_CREDENTIALS_BLOCK = "dojo-aim-artifacts"


class S3ArtifactsLocation(S3Config):
    """dojo's S3Config without the keys, which come from an AwsCredentials block instead.

    Flow parameters are stored with the run and shown in the UI, so keys must never be one.
    """
    accesskey: SkipJsonSchema[Optional[str]] = None
    secretkey: SkipJsonSchema[Optional[str]] = None

    @field_validator("accesskey", "secretkey")
    @classmethod
    def no_keys(cls, value):
        if value:
            raise ValueError(
                "Don't pass S3 keys as parameters: they are stored with the run and shown in the UI. "
                "Put them in an AWS Credentials block and name it in artifacts_credentials_block."
            )
        return value


class DojoLoggerConfig(AimLoggerConfig):
    artifacts_location: Optional[Union[str, S3ArtifactsLocation]] = Field(None,
        description="Aim artifacts location: a local path, or an S3 bucket whose keys come from "
                    "artifacts_credentials_block")


class DojoTrainingRunConfig(TrainingRunConfig):
    logger: DojoLoggerConfig


def _logger_config_with_keys(logger_config: DojoLoggerConfig, credentials_block: str) -> AimLoggerConfig:
    """The logger config for the container, with S3 keys filled in from the block."""
    location = logger_config.artifacts_location
    if not isinstance(location, S3Config):
        return logger_config
    from prefect_aws import AwsCredentials

    credentials = AwsCredentials.load(credentials_block)
    if not (credentials.aws_access_key_id and credentials.aws_secret_access_key):
        raise ValueError(f"AWS Credentials block {credentials_block!r} has no access key or secret key")
    keyed = S3Config(**location.model_dump(exclude={"accesskey", "secretkey"}),
                     accesskey=credentials.aws_access_key_id,
                     secretkey=credentials.aws_secret_access_key.get_secret_value())
    # dojo's own model, whose field takes an S3Config with keys.
    return AimLoggerConfig(**logger_config.model_dump(exclude={"artifacts_location"}), artifacts_location=keyed)

class VolumeMapping(BaseModel):
    host_path: str = Field(..., description="Path on the host machine")
    container_path: str = Field(..., description="Path inside the container")
    mode: Literal['ro','rw']= Field('rw', description="Mount mode: read-only ('ro') or read-write ('rw')")

@task()
def run_dojo_container(output_dir: str, input_volumes: List[VolumeMapping], subcommands:List[str], training_run_config: DojoTrainingRunConfig, device_ids:List[str]=['all'], artifacts_credentials_block: str = ARTIFACTS_CREDENTIALS_BLOCK):
    """
    Run Image Classifier Dojo in a Docker container.

    ``device_ids`` is a GPU spec (see src/tasks/gpu.py); ['all'] means every idle GPU.
    S3 keys for an S3 ``artifacts_location`` come from the AwsCredentials block
    ``artifacts_credentials_block``. dojo only reads them from its command line,
    so they are passed there, but never logged.
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

    def command_with(logger_config):
        return subcommands + [
            "--logger", logger_config.model_dump_json(),
            "--dataset_config", training_run_config.dataset_config.model_dump_json(),
            "--model", training_run_config.model.model_dump_json(),
            "--training", training_run_config.training.model_dump_json(),
            "--runtime", training_run_config.runtime.model_dump_json(),
        ]

    # Log the command without keys; the container gets the one with them.
    logger.info(" ".join(command_with(training_run_config.logger)))
    command = command_with(_logger_config_with_keys(training_run_config.logger, artifacts_credentials_block))
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
