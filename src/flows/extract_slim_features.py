from prefect import flow, get_run_logger
import os

from src.params.params_extract_slim_features import ExtractSlimFeaturesParams
from src.tasks.pull_images import pull_images
from src.tasks.run_extract_slim_features import run_extract_slim_features


@flow(name="Extract Slim Features", log_prints=True)
def extract_slim_features_flow(extract_features_params: ExtractSlimFeaturesParams):
    """
    Flow for extracting slim features from IFCB data.
    
    This flow:
    1. Creates the output directory if it doesn't exist
    2. Pulls the latest Docker image for feature extraction
    3. Runs extract_slim_features.py in a Docker container
    4. Extracts features and saves them as CSV files
    5. Saves blob images as ZIP files
    """
    
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    os.makedirs(extract_features_params.output_directory, exist_ok=True)
    logger.info(f"Output directory: {extract_features_params.output_directory}")
    
    # Define the Docker image (leave blank for now as requested)
    extract_features_image = "ghcr.io/whoigit/ifcb-features:docker"
    
    # Pull the latest image if specified
    if extract_features_image:
        pull_images([extract_features_image])
    else:
        logger.warning("No Docker image specified - skipping image pull")
    
    # Log processing details
    if extract_features_params.bins:
        logger.info(f"Processing {len(extract_features_params.bins)} specific bins: {extract_features_params.bins}")
    else:
        logger.info("Processing all bins in the data directory")
    
    logger.info(f"Data directory: {extract_features_params.data_directory}")
    
    # Run feature extraction
    run_extract_slim_features(extract_features_params, extract_features_image)
    
    logger.info("Feature extraction completed successfully")


if __name__ == "__main__":
    extract_slim_features_flow.serve(
        name="extract-slim-features",
        tags=["feature-extraction", "ifcb", "roi-features"],
    )
