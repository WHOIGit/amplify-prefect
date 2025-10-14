from prefect import flow, get_run_logger
import os

from src.params.params_feature_validation import FeatureValidationParams
from src.tasks.pull_images import pull_images
from src.tasks.run_feature_validation import run_feature_validation
from src.tasks.run_blob_comparison import run_blob_comparison
from src.tasks.create_combined_validation_report import create_combined_validation_report


@flow(name="IFCB Feature Validation", log_prints=True)
def feature_validation_flow(validation_params: FeatureValidationParams):
    """
    Flow for validating IFCB feature extraction against ground truth.

    This flow:
    1. Creates the output directory if it doesn't exist
    2. Pulls the latest Docker image for validation
    3. Runs validation comparing predicted vs ground truth features from VastDB
    4. Optionally runs pixel-by-pixel blob comparison
    5. Creates a comprehensive validation report combining all results
    """

    logger = get_run_logger()

    # Create output directory if it doesn't exist
    os.makedirs(validation_params.output_directory, exist_ok=True)
    logger.info(f"Output directory: {validation_params.output_directory}")

    # Pull the validation image
    pull_images([validation_params.validation_image])

    # Log validation configuration
    logger.info(f"Validation image: {validation_params.validation_image}")
    logger.info(f"Predicted features: {validation_params.pred_bucket}.{validation_params.pred_schema}.{validation_params.pred_table}")
    logger.info(f"Ground truth features: {validation_params.gt_bucket}.{validation_params.gt_schema}.{validation_params.gt_table}")

    if validation_params.sample_ids:
        logger.info(f"Validating {len(validation_params.sample_ids)} specific samples")
    else:
        logger.info("Validating all samples in predicted features table")

    # Run numeric feature validation
    run_feature_validation(validation_params)

    logger.info("✓ Feature validation completed successfully")

    # Run blob comparison if enabled
    if validation_params.enable_blob_comparison:
        logger.info("Starting blob comparison...")
        run_blob_comparison(validation_params)
        logger.info("✓ Blob comparison completed successfully")

    # Create comprehensive combined report
    logger.info("Creating comprehensive validation report...")
    create_combined_validation_report(validation_params)

    logger.info("✓ All validation tasks completed successfully")


if __name__ == "__main__":
    feature_validation_flow.serve(
        name="ifcb-feature-validation",
        tags=["validation", "ifcb", "features", "metrics"],
    )
