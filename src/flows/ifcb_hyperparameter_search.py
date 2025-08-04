from prefect import flow

from src.params.params_ifcb_hyperparameter_search import IFCBHyperparameterSearchParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_hyperparameter_search import run_ifcb_hyperparameter_search


@flow(name="IFCB Hyperparameter Search")
def ifcb_hyperparameter_search_flow(search_params: IFCBHyperparameterSearchParams):
    """
    Flow for hyperparameter search of IFCB flow metric anomaly detection models.
    
    This flow:
    1. Pulls the latest IFCB flow metric Docker image
    2. Runs hyperparameter search by training models with different parameter combinations
    3. Saves results in separate subdirectories under the base output directory
    """
    
    # Define the Docker image for IFCB flow metric training
    ifcb_image = "ghcr.io/whoigit/ifcb-flow-metric:feature-isolation"
    
    # Pull the latest image
    pull_images([ifcb_image])
    
    # Run hyperparameter search
    results = run_ifcb_hyperparameter_search(search_params)
    
    return results


if __name__ == "__main__":
    ifcb_hyperparameter_search_flow.serve(
        name="ifcb-hyperparameter-search",
        tags=["training", "ifcb", "anomaly-detection", "hyperparameter-search"],
    )
