from prefect import flow, get_run_logger

from src.params.params_ifcb_flow_metric import IFCBInferenceParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_flow_metric_inference import run_ifcb_flow_metric_inference
from src.utils.output_dir_utils import create_output_dir


@flow(name="IFCB Flow Metric Inference")
def ifcb_inference_flow(ifcb_inference_params: IFCBInferenceParams):
    """
    Flow for running IFCB flow metric inference/scoring.
    
    This flow:
    1. Pulls the latest IFCB flow metric Docker image
    2. Runs inference/scoring in a containerized environment
    """
    
    # Define the Docker image for IFCB flow metric inference
    ifcb_image = "ghcr.io/whoigit/ifcb-flow-metric:main"
    
    # Pull the latest image
    pull_images([ifcb_image])

    # The container runs as this user, so the output directory must exist
    # first; otherwise Docker creates it owned by root.
    create_output_dir(ifcb_inference_params.output_dir, get_run_logger())

    # Run IFCB flow metric inference
    run_ifcb_flow_metric_inference(ifcb_inference_params, ifcb_image)


if __name__ == "__main__":
    ifcb_inference_flow.serve(
        name="ifcb-flow-metric-inference",
        tags=["inference", "ifcb", "anomaly-detection"],
    )
