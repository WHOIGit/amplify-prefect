from prefect import task
import docker
import os

from prefect import get_run_logger
from prefect_aws import AwsCredentials

from src.prov import on_task_complete
from src.params.params_extract_slim_features import ExtractSlimFeaturesParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_extract_slim_features(extract_features_params: ExtractSlimFeaturesParams):
    """
    Run extract_slim_features.py in a Docker container to extract IFCB features.
    """

    client = docker.from_env()
    logger = get_run_logger()

    # Load AWS credentials from Prefect block
    aws_credentials = AwsCredentials.load("vasts3-creds")

    # Set up volumes
    volumes = {
        extract_features_params.data_directory: {'bind': '/app/data', 'mode': 'ro'},
        extract_features_params.output_directory: {'bind': '/app/output', 'mode': 'rw'}
    }

    # Set up environment variables for AWS credentials
    environment = {
        'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
        'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
    }

    # Build command arguments (ENTRYPOINT already includes "python extract_slim_features.py")
    command_args = [
        "/app/data",
        "/app/output"
    ]

    # Add blob storage arguments
    command_args.extend([
        "--blob-storage-mode", extract_features_params.blob_storage_mode
    ])

    if extract_features_params.blob_storage_mode == "s3":
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
        command_args.extend([
            "--vastdb-bucket", extract_features_params.vastdb_bucket,
            "--vastdb-schema", extract_features_params.vastdb_schema,
            "--vastdb-table", extract_features_params.vastdb_table,
            "--vastdb-url", vastdb_url
        ])

    # Add optional bins flag if provided
    if extract_features_params.bins is not None and len(extract_features_params.bins) > 0:
        command_args.extend(["--bins"] + extract_features_params.bins)

    # Add GPU batch processing arguments if enabled
    if extract_features_params.batch_processing:
        command_args.extend([
            "--batch-processing",
            "--min-batch-size", str(extract_features_params.min_batch_size),
            "--max-batch-size", str(extract_features_params.max_batch_size)
        ])
        if extract_features_params.gpu_device is not None:
            command_args.extend(["--gpu-device", str(extract_features_params.gpu_device)])

    logger.info(f'Running container with command: {" ".join(command_args)}')

    # Configure GPU device requests if batch processing is enabled
    device_requests = []
    if extract_features_params.batch_processing:
        if extract_features_params.gpu_device is not None:
            # Request specific GPU device
            device_requests = [docker.types.DeviceRequest(
                device_ids=[str(extract_features_params.gpu_device)],
                capabilities=[["gpu"]]
            )]
        else:
            # Request all available GPUs
            device_requests = [docker.types.DeviceRequest(
                device_ids=["all"],
                capabilities=[["gpu"]]
            )]
        logger.info(f"GPU device requests configured: {device_requests}")

    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()

        container = client.containers.run(
            extract_features_params.extract_features_image,
            command_args,
            volumes=volumes,
            environment=environment,
            user=f"{uid}:{gid}",
            device_requests=device_requests if device_requests else None,
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
