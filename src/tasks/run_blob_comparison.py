from prefect import task
import docker
import os
import json
import pandas as pd
from pathlib import Path

from prefect import get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_image_artifact
from prefect_aws import AwsCredentials

from src.prov import on_task_complete
from src.params.params_feature_validation import FeatureValidationParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_blob_comparison(validation_params: FeatureValidationParams):
    """
    Run IFCB blob comparison in a Docker container.

    Compares predicted blobs against ground truth blobs pixel-by-pixel and creates
    Prefect artifacts with comparison results including visualizations.
    """

    client = docker.from_env()
    logger = get_run_logger()

    # Skip if blob comparison is disabled
    if not validation_params.enable_blob_comparison:
        logger.info("Blob comparison is disabled, skipping")
        return

    # Load AWS credentials from Prefect block
    aws_credentials = AwsCredentials.load("vasts3-creds")

    # Set up volumes - mount output directory
    blob_output_dir = os.path.join(validation_params.output_directory, "blob_comparison")
    os.makedirs(blob_output_dir, exist_ok=True)

    volumes = {
        blob_output_dir: {'bind': '/app/blob_comparison_output', 'mode': 'rw'}
    }

    # Set up environment variables for S3 credentials
    environment = {
        'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
        'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
        'PYTHONUNBUFFERED': '1',
    }

    # Build command arguments
    command_args = [
        "compare_blobs.py",
        "--pred-bucket", validation_params.blob_pred_bucket,
        "--gt-bucket", validation_params.blob_gt_bucket,
        "--s3-url", validation_params.blob_s3_url,
        "--pred-prefix", validation_params.blob_pred_prefix,
        "--gt-prefix", validation_params.blob_gt_prefix,
        "--output-dir", "/app/blob_comparison_output",
        "--top-n-worst", str(validation_params.blob_top_n_worst),
    ]

    # Add optional sample IDs filter
    if validation_params.sample_ids:
        command_args.extend(["--sample-ids"] + validation_params.sample_ids)

    logger.info(f'Running blob comparison container with command: {" ".join(command_args)}')

    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()

        # Run container in detached mode to stream logs properly
        container = client.containers.run(
            validation_params.validation_image,
            command_args,
            volumes=volumes,
            environment=environment,
            user=f"{uid}:{gid}",
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

        logger.info("✓ Blob comparison completed successfully")

        # Read results and create Prefect artifacts
        results_file = os.path.join(blob_output_dir, "blob_comparison_results.json")
        csv_file = os.path.join(blob_output_dir, "blob_comparison_details.csv")

        # Create summary markdown artifact
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            summary_stats = results['summary_stats']

            markdown_content = f"""# IFCB Blob Comparison Summary

## Overall Statistics
- **Total blobs compared**: {summary_stats['total_blobs']}
- **Mean IoU**: {summary_stats['mean_iou']:.4f}
- **Median IoU**: {summary_stats['median_iou']:.4f}
- **Std Dev IoU**: {summary_stats['std_iou']:.4f}
- **Mean Dice coefficient**: {summary_stats['mean_dice']:.4f}
- **Median Dice coefficient**: {summary_stats['median_dice']:.4f}
- **Mean accuracy**: {summary_stats['mean_accuracy']:.4f}

## Quality Metrics
- **Perfect matches (IoU=1.0)**: {summary_stats['perfect_matches']} ({100*summary_stats['perfect_matches']/summary_stats['total_blobs']:.1f}%)
- **Near-perfect matches (IoU≥0.95)**: {summary_stats['near_perfect_matches']} ({100*summary_stats['near_perfect_matches']/summary_stats['total_blobs']:.1f}%)
- **Poor matches (IoU<0.5)**: {summary_stats['poor_matches']} ({100*summary_stats['poor_matches']/summary_stats['total_blobs']:.1f}%)

## Configuration
- **Predicted Blobs**: `s3://{validation_params.blob_pred_bucket}/{validation_params.blob_pred_prefix}`
- **Ground Truth Blobs**: `s3://{validation_params.blob_gt_bucket}/{validation_params.blob_gt_prefix}`
- **Visualizations**: Top {validation_params.blob_top_n_worst} worst cases shown below
"""

            create_markdown_artifact(
                key="blob-comparison-summary",
                markdown=markdown_content,
                description="IFCB Blob Comparison Summary"
            )

        # Create detailed metrics table artifact
        if os.path.exists(csv_file):
            metrics_df = pd.read_csv(csv_file)

            # Create table showing worst matches
            worst_matches = metrics_df.nsmallest(10, 'iou')[['sample_id', 'roi_number', 'iou', 'dice', 'accuracy', 'diff_pixels']]

            create_table_artifact(
                key="worst-blob-matches",
                table=worst_matches.to_dict('records'),
                description="Top 10 Worst Blob Matches by IoU"
            )

            # Create table showing best matches
            best_matches = metrics_df.nlargest(10, 'iou')[['sample_id', 'roi_number', 'iou', 'dice', 'accuracy', 'diff_pixels']]

            create_table_artifact(
                key="best-blob-matches",
                table=best_matches.to_dict('records'),
                description="Top 10 Best Blob Matches by IoU"
            )

        # Create image artifacts for worst cases
        comparison_images_dir = os.path.join(blob_output_dir, "blob_comparisons")
        if os.path.exists(comparison_images_dir):
            image_files = sorted(Path(comparison_images_dir).glob("*.png"))

            for idx, image_path in enumerate(image_files[:validation_params.blob_top_n_worst]):
                # Extract info from filename: {sample_id}_{roi_number:05d}_iou{iou:.3f}.png
                filename = image_path.stem
                parts = filename.rsplit('_iou', 1)
                sample_roi = parts[0] if len(parts) > 0 else filename
                iou_str = parts[1] if len(parts) > 1 else "unknown"

                with open(image_path, 'rb') as f:
                    image_data = f.read()

                create_image_artifact(
                    key=f"blob-comparison-{idx+1}",
                    image=image_data,
                    description=f"Blob comparison: {sample_roi} (IoU={iou_str})"
                )

        logger.info(f"✓ Created Prefect artifacts with blob comparison results")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
