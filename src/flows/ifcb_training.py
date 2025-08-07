from prefect import flow

from src.params.params_ifcb_flow_metric import IFCBTrainingParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_training import run_ifcb_training


@flow(name="IFCB Flow Metric Training")
def ifcb_training_flow(ifcb_training_params: IFCBTrainingParams):
    """
    Flow for training IFCB flow metric anomaly detection models.
    
    This flow:
    1. Pulls the latest IFCB flow metric Docker image
    2. Runs model training in a containerized environment
    """
    
    # Define the Docker image for IFCB flow metric training
    ifcb_image = "ghcr.io/whoigit/ifcb-flow-metric:main"
    
    # Pull the latest image
    pull_images([ifcb_image])
    
    # Run IFCB flow metric training
    run_ifcb_training(ifcb_training_params, ifcb_image)


if __name__ == "__main__":
    ifcb_training_flow.serve(
        name="ifcb-flow-metric-training",
        tags=["training", "ifcb", "anomaly-detection"],
    )
