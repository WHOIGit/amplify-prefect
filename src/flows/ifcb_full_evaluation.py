from prefect import flow
import os

from src.params.params_ifcb_flow_metric import IFCBFullEvaluationParams, IFCBInferenceParams, IFCBEvaluationParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_flow_metric_inference import run_ifcb_flow_metric_inference
from src.tasks.run_ifcb_flow_metric_evaluation import run_ifcb_flow_metric_evaluation


@flow(name="IFCB Flow Metric Full Evaluation")
def ifcb_full_evaluation_flow(ifcb_full_evaluation_params: IFCBFullEvaluationParams):
    """
    Flow for full IFCB flow metric evaluation comparing known bad vs normal data.
    
    This flow:
    1. Finds all subdirectories in the bad data directory
    2. For each subdirectory:
       - Runs inference on that subdirectory and saves CSV
       - Runs inference on normal data and saves CSV  
       - Creates violin plot comparing the two distributions
    """
    
    # Define the Docker image
    ifcb_image = "ghcr.io/whoigit/ifcb-flow-metric:main"
    
    # Pull the latest image
    pull_images([ifcb_image])
    
    # Get all subdirectories in bad data directory
    bad_subdirs = [d for d in os.listdir(ifcb_full_evaluation_params.bad_data_dir) 
                   if os.path.isdir(os.path.join(ifcb_full_evaluation_params.bad_data_dir, d))]
    
    print(f"Found {len(bad_subdirs)} bad data subdirectories: {bad_subdirs}")
    
    # Run inference on normal data once (will be reused for all comparisons)
    normal_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, "normal_data_scores.csv")
    normal_inference_params = IFCBInferenceParams(
        data_dir=ifcb_full_evaluation_params.normal_data_dir,
        output_dir=ifcb_full_evaluation_params.output_dir,
        model_path=ifcb_full_evaluation_params.model_path,
        n_jobs=ifcb_full_evaluation_params.n_jobs,
        aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
        chunk_size=ifcb_full_evaluation_params.chunk_size,
        output_filename="normal_data_scores.csv"
    )
    
    run_ifcb_flow_metric_inference(normal_inference_params, ifcb_image)
    
    # Process each bad data subdirectory
    for subdir_name in bad_subdirs:
        print(f"Processing bad data subdirectory: {subdir_name}")
        
        subdir_path = os.path.join(ifcb_full_evaluation_params.bad_data_dir, subdir_name)
        
        # Run inference on this bad data subdirectory
        bad_csv_filename = f"bad_data_{subdir_name}_scores.csv"
        bad_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_csv_filename)
        
        bad_inference_params = IFCBInferenceParams(
            data_dir=subdir_path,
            output_dir=ifcb_full_evaluation_params.output_dir,
            model_path=ifcb_full_evaluation_params.model_path,
            n_jobs=ifcb_full_evaluation_params.n_jobs,
            aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
            chunk_size=ifcb_full_evaluation_params.chunk_size,
            output_filename=bad_csv_filename
        )
        
        run_ifcb_flow_metric_inference(bad_inference_params, ifcb_image)
        
        # Create violin plot comparing this bad subdirectory to normal data
        plot_filename = f"evaluation_{subdir_name}_vs_{ifcb_full_evaluation_params.normal_data_name.replace(' ', '_').lower()}.png"
        plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix}: {subdir_name} vs {ifcb_full_evaluation_params.normal_data_name}"
        
        evaluation_params = IFCBEvaluationParams(
            csv1_path=bad_csv_path,
            csv2_path=normal_csv_path,
            output_dir=ifcb_full_evaluation_params.output_dir,
            output_filename=plot_filename,
            title=plot_title,
            name1=subdir_name,
            name2=ifcb_full_evaluation_params.normal_data_name
        )
        
        run_ifcb_flow_metric_evaluation(evaluation_params, ifcb_image)
        
        print(f"Completed evaluation for {subdir_name}")
    
    print(f"Full evaluation completed for {len(bad_subdirs)} bad data categories")


if __name__ == "__main__":
    ifcb_full_evaluation_flow.serve(
        name="ifcb-flow-metric-full-evaluation",
        tags=["evaluation", "ifcb", "anomaly-detection", "comparison"],
    )
