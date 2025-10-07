from prefect import task
import docker
import os
import json
import pandas as pd

from prefect import get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
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

        # Read results and create Prefect artifacts
        local_output_path = os.path.join(validation_params.output_directory, validation_params.output_filename)
        local_summary_path = os.path.join(validation_params.output_directory, validation_params.summary_filename)

        # Create summary markdown artifact
        if os.path.exists(local_summary_path):
            with open(local_summary_path, 'r') as f:
                summary = json.load(f)

            markdown_content = f"""# IFCB Feature Validation Summary

## Overall Statistics
- **Total features compared**: {summary['total_features']}
- **Mean RMSE**: {summary['mean_rmse']:.4f}
- **Median RMSE**: {summary['median_rmse']:.4f}
- **Mean MAE**: {summary['mean_mae']:.4f}
- **Median MAE**: {summary['median_mae']:.4f}
- **Mean R²**: {summary['mean_r2']:.4f}
- **Median R²**: {summary['median_r2']:.4f}
- **Mean Pearson correlation**: {summary['mean_pearson_r']:.4f}
- **Median Pearson correlation**: {summary['median_pearson_r']:.4f}

## Quality Metrics
- **Features with high correlation (r > 0.9)**: {summary['features_with_high_correlation']}
- **Features with low R² (< 0.5)**: {summary['features_with_low_r2']}

## Configuration
- **Predicted**: `{validation_params.pred_bucket}.{validation_params.pred_schema}.{validation_params.pred_table}`
- **Ground Truth**: `{validation_params.gt_bucket}.{validation_params.gt_schema}.{validation_params.gt_table}`
"""

            create_markdown_artifact(
                key="validation-summary",
                markdown=markdown_content,
                description="IFCB Feature Validation Summary"
            )

        # Create detailed metrics table artifact
        if os.path.exists(local_output_path):
            metrics_df = pd.read_csv(local_output_path)

            # Create table showing top features by different metrics
            top_by_r2 = metrics_df.nlargest(10, 'r2')[['feature', 'r2', 'rmse', 'mae', 'pearson_r']]

            create_table_artifact(
                key="top-features-by-r2",
                table=top_by_r2.to_dict('records'),
                description="Top 10 Features by R² Score"
            )

            # Create table showing worst features
            worst_by_r2 = metrics_df.nsmallest(10, 'r2')[['feature', 'r2', 'rmse', 'mae', 'pearson_r']]

            create_table_artifact(
                key="worst-features-by-r2",
                table=worst_by_r2.to_dict('records'),
                description="Bottom 10 Features by R² Score"
            )

        logger.info(f"✓ Created Prefect artifacts with validation results")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
