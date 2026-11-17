from prefect import task
import docker
import os

from prefect import get_run_logger

from src.params.params_extract_slim_features import ExtractSlimFeaturesParams, SlimFeaturesSource
from src.tasks.gpu import allocate_gpus, run_container


MAIN_SLIM_FEATURES_IMAGE = "ghcr.io/whoigit/ifcb-features:latest"


def resolve_extract_slim_features_image(extract_features_params: ExtractSlimFeaturesParams) -> str:
    if extract_features_params.extract_features_source == SlimFeaturesSource.main:
        return MAIN_SLIM_FEATURES_IMAGE

    if not extract_features_params.extract_features_image:
        raise ValueError("extract_features_image is required when extract_features_source='storage'")

    return extract_features_params.extract_features_image


def build_extract_slim_features_command(
    extract_features_params: ExtractSlimFeaturesParams,
    container_gpu_device: int | None = None,
) -> list[str]:
    """``container_gpu_device`` overrides gpu_device with the GPU's index inside the container."""
    command_args = [
        "/app/data",
        "/app/output"
    ]

    if extract_features_params.extract_features_source == SlimFeaturesSource.main:
        command_args.extend(["--workers", str(extract_features_params.workers)])
        if extract_features_params.bins:
            command_args.extend(["--bins"] + extract_features_params.bins)
        return command_args

    # Add blob storage arguments
    command_args.extend([
        "--blob-storage-mode", extract_features_params.blob_storage_mode
    ])

    if extract_features_params.blob_storage_mode == "s3":
        if not extract_features_params.s3_bucket or not extract_features_params.s3_url:
            raise ValueError("s3_bucket and s3_url are required when blob_storage_mode='s3'")

        command_args.extend([
            "--s3-bucket", extract_features_params.s3_bucket,
            "--s3-url", extract_features_params.s3_url,
            "--s3-prefix", extract_features_params.s3_prefix
        ])

    # Add feature storage arguments
    command_args.extend([
        "--feature-storage-mode", extract_features_params.feature_storage_mode
    ])

    if extract_features_params.feature_storage_mode == "vastdb":
        vastdb_url = extract_features_params.vastdb_url or extract_features_params.s3_url
        if (
            not extract_features_params.vastdb_bucket
            or not extract_features_params.vastdb_schema
            or not extract_features_params.vastdb_table
            or not vastdb_url
        ):
            raise ValueError(
                "vastdb_bucket, vastdb_schema, vastdb_table, and vastdb_url or s3_url "
                "are required when feature_storage_mode='vastdb'"
            )

        command_args.extend([
            "--vastdb-bucket", extract_features_params.vastdb_bucket,
            "--vastdb-schema", extract_features_params.vastdb_schema,
            "--vastdb-table", extract_features_params.vastdb_table,
            "--vastdb-url", vastdb_url
        ])

    # Add optional bins flag if provided
    if extract_features_params.bins:
        command_args.extend(["--bins"] + extract_features_params.bins)

    # Add GPU batch processing arguments if enabled
    if (
        extract_features_params.extract_features_source == SlimFeaturesSource.storage
        and extract_features_params.batch_processing
    ):
        command_args.extend([
            "--batch-processing",
            "--min-batch-size", str(extract_features_params.min_batch_size),
            "--max-batch-size", str(extract_features_params.max_batch_size)
        ])
        gpu_device = (
            container_gpu_device if container_gpu_device is not None else extract_features_params.gpu_device
        )
        if gpu_device is not None:
            command_args.extend(["--gpu-device", str(gpu_device)])

    return command_args


@task(log_prints=True)
def run_extract_slim_features(extract_features_params: ExtractSlimFeaturesParams):
    """
    Run IFCB feature extraction in a Docker container.

    The main image runs extract_features_batch.py, which writes the MATLAB-compatible
    day-based layout under the output directory; the storage image is unchanged.
    """

    client = docker.from_env()
    logger = get_run_logger()
    extract_features_image = resolve_extract_slim_features_image(extract_features_params)

    # Set up volumes
    volumes = {
        extract_features_params.data_directory: {'bind': '/app/data', 'mode': 'ro'},
        extract_features_params.output_directory: {'bind': '/app/output', 'mode': 'rw'}
    }

    # Set up environment variables
    environment = {
        'PYTHONUNBUFFERED': '1',  # Disable Python output buffering for real-time logs
    }

    if extract_features_params.extract_features_source == SlimFeaturesSource.storage:
        from prefect_aws import AwsCredentials

        # Load AWS credentials from Prefect block for the storage-capable image.
        aws_credentials = AwsCredentials.load("vasts3-creds")
        environment.update({
            'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
        })

    uses_gpu = (
        extract_features_params.extract_features_source == SlimFeaturesSource.storage
        and extract_features_params.batch_processing
    )
    # Batch processing uses one GPU: the requested one, or any idle one.
    gpu_spec = "cpu"
    if uses_gpu:
        gpu_spec = extract_features_params.gpu_device if extract_features_params.gpu_device is not None else "auto:1"

    with allocate_gpus(gpu_spec, logger) as gpus:
        # The container only sees the allocated GPU, as device 0.
        command_args = build_extract_slim_features_command(
            extract_features_params, container_gpu_device=0 if gpus else None
        )
        logger.info(f'Running container with command: {" ".join(command_args)}')
        logger.info(f'Using Docker image: {extract_features_image}')

        # Run as the current user so output files are owned by them.
        run_container(
            client,
            extract_features_image,
            command_args,
            logger,
            gpus=gpus,
            volumes=volumes,
            environment=environment,
            user=f"{os.getuid()}:{os.getgid()}",
        )
