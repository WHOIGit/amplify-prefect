from prefect import task
import docker
import os

from prefect import get_run_logger
from prefect_aws import AwsCredentials

from src.prov import on_task_complete
from src.params.params_feature_validation import FeatureValidationParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_feature_validation(validation_params: FeatureValidationParams):
    """
    Run IFCB feature validation in a Docker container.

    Compares predicted features against ground truth and creates Prefect artifacts
    with validation results.
    """

    client = docker.from_env()
    logger = get_run_logger()

    # Load AWS credentials from Prefect block
    aws_credentials = AwsCredentials.load("vasts3-creds")

    # Set up volumes - mount output directory
    volumes = {
        validation_params.output_directory: {'bind': '/app/output', 'mode': 'rw'}
    }

    # Set up environment variables for VastDB credentials
    environment = {
        'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
        'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
        'AWS_VASTDB_URL': validation_params.vastdb_url,
    }

    # Build command arguments
    output_path = f"/app/output/{validation_params.output_filename}"
    summary_path = f"/app/output/{validation_params.summary_filename}"

    command_args = [
        "validate_features.py",
        "--pred-bucket", validation_params.pred_bucket,
        "--pred-schema", validation_params.pred_schema,
        "--pred-table", validation_params.pred_table,
        "--gt-bucket", validation_params.gt_bucket,
        "--gt-schema", validation_params.gt_schema,
        "--gt-table", validation_params.gt_table,
        "--output", output_path,
        "--summary", summary_path,
        "--pred-sample-col", validation_params.pred_sample_col,
        "--gt-sample-col", validation_params.gt_sample_col,
        "--pred-roi-col", validation_params.pred_roi_col,
        "--gt-roi-col", validation_params.gt_roi_col,
    ]

    # Add optional sample IDs filter
    if validation_params.sample_ids:
        command_args.extend(["--sample-ids"] + validation_params.sample_ids)

    logger.info(f'Running validation container with command: {" ".join(command_args)}')

    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()

        container = client.containers.run(
            validation_params.validation_image,
            command_args,
            volumes=volumes,
            environment=environment,
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

        logger.info("✓ Validation completed successfully")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
