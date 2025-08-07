from prefect import flow

from src.params.params_ifcb_flow_metric import IFCBInferenceParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_flow_metric_inference import run_ifcb_flow_metric_inference


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
    
    # Run IFCB flow metric inference
    run_ifcb_flow_metric_inference(ifcb_inference_params, ifcb_image)


if __name__ == "__main__":
    ifcb_inference_flow.serve(
        name="ifcb-flow-metric-inference",
        tags=["inference", "ifcb", "anomaly-detection"],
    )
