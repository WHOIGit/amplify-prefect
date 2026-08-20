from prefect import flow, get_run_logger
import os

from src.params.params_extract_slim_features import ExtractSlimFeaturesParams
from src.tasks.pull_images import pull_images
from src.tasks.run_extract_slim_features import resolve_extract_slim_features_image, run_extract_slim_features


@flow(name="Extract Slim Features", log_prints=True)
def extract_slim_features_flow(extract_features_params: ExtractSlimFeaturesParams):
    """
    Flow for extracting slim features from IFCB data.

    This flow:
    1. Creates the output directory if it doesn't exist (for local storage modes)
    2. Pulls the latest Docker image for feature extraction
    3. Runs extract_slim_features.py in a Docker container
    4. Extracts features through either the storage-capable image or the main-branch image
    """
    
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist.
    #
    # makedirs' own mode= is masked by the umask, so under the Prefect process
    # umask (0022) the root would land 0755: setgid inherited from the parent
    # gives it the right group, but the group cannot write to it. Everything the
    # container creates underneath is 0775 (the image sets umask 0002), so
    # without this the root is the one directory the group can't add to.
    os.makedirs(extract_features_params.output_directory, exist_ok=True)
    try:
        os.chmod(extract_features_params.output_directory, 0o2775)
    except OSError as e:
        # A pre-existing directory owned by another user cannot be chmod'ed;
        # that is not worth failing an extraction run over.
        logger.warning(
            f"Could not set 2775 on {extract_features_params.output_directory}: {e}"
        )
    logger.info(f"Output directory: {extract_features_params.output_directory}")

    extract_features_image = resolve_extract_slim_features_image(extract_features_params)

    # Pull the latest image
    logger.info(f"Using Docker image: {extract_features_image}")
    pull_images([extract_features_image])
    
    # Log processing details
    if extract_features_params.bins:
        logger.info(f"Processing {len(extract_features_params.bins)} specific bins: {extract_features_params.bins}")
    else:
        logger.info("Processing all bins in the data directory")
    
    logger.info(f"Data directory: {extract_features_params.data_directory}")
    
    # Run feature extraction
    run_extract_slim_features(extract_features_params)
    
    logger.info("Feature extraction completed successfully")


if __name__ == "__main__":
    extract_slim_features_flow.serve(
        name="extract-slim-features",
        tags=["feature-extraction", "ifcb", "roi-features"],
    )
