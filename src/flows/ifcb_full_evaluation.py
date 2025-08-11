from prefect import flow
import os

from src.params.params_ifcb_flow_metric import IFCBFullEvaluationParams, IFCBInferenceParams, IFCBEvaluationParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_flow_metric_inference import run_ifcb_flow_metric_inference
from src.tasks.run_ifcb_flow_metric_evaluation import run_ifcb_flow_metric_evaluation
from src.tasks.merge_csv_files import merge_csv_files
from src.utils.bin_utils import create_bin_type_id_file


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
    
    # Run inference on normal data with separate I and D models
    normal_i_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, "normal_data_i_bins_scores.csv")
    normal_d_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, "normal_data_d_bins_scores.csv")
    
    print("Running inference on normal data with separate I and D models...")
    
    # Create temporary ID files for I and D bins
    temp_i_file, normal_i_bins = create_bin_type_id_file(ifcb_full_evaluation_params.normal_data_dir, "I")
    temp_d_file, normal_d_bins = create_bin_type_id_file(ifcb_full_evaluation_params.normal_data_dir, "D")
    
    print(f"Found {normal_i_bins} I bins and {normal_d_bins} D bins in normal data")
    
    try:
        # Run inference on I bins if any exist
        if normal_i_bins > 0:
            print(f"Processing {normal_i_bins} normal I bins with I model...")
            normal_i_inference_params = IFCBInferenceParams(
                data_dir=ifcb_full_evaluation_params.normal_data_dir,
                output_dir=ifcb_full_evaluation_params.output_dir,
                model_path=ifcb_full_evaluation_params.i_model_path,
                id_file=temp_i_file,
                n_jobs=ifcb_full_evaluation_params.n_jobs,
                aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                chunk_size=ifcb_full_evaluation_params.chunk_size,
                output_filename="normal_data_i_bins_scores.csv"
            )
            run_ifcb_flow_metric_inference(normal_i_inference_params, ifcb_image)
        
        # Run inference on D bins if any exist
        if normal_d_bins > 0:
            print(f"Processing {normal_d_bins} normal D bins with D model...")
            normal_d_inference_params = IFCBInferenceParams(
                data_dir=ifcb_full_evaluation_params.normal_data_dir,
                output_dir=ifcb_full_evaluation_params.output_dir,
                model_path=ifcb_full_evaluation_params.d_model_path,
                id_file=temp_d_file,
                n_jobs=ifcb_full_evaluation_params.n_jobs,
                aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                chunk_size=ifcb_full_evaluation_params.chunk_size,
                output_filename="normal_data_d_bins_scores.csv"
            )
            run_ifcb_flow_metric_inference(normal_d_inference_params, ifcb_image)
            
    finally:
        # Clean up temporary files
        if temp_i_file and os.path.exists(temp_i_file):
            os.unlink(temp_i_file)
        if temp_d_file and os.path.exists(temp_d_file):
            os.unlink(temp_d_file)
    
    print(f"Normal data inference completed: {normal_i_bins} I bins, {normal_d_bins} D bins")
    
    # Process each bad data subdirectory
    for subdir_name in bad_subdirs:
        print(f"Processing bad data subdirectory: {subdir_name}")
        
        subdir_path = os.path.join(ifcb_full_evaluation_params.bad_data_dir, subdir_name)
        
        # Run inference on this bad data subdirectory with separate I and D models
        bad_i_csv_filename = f"bad_data_{subdir_name}_i_bins_scores.csv"
        bad_d_csv_filename = f"bad_data_{subdir_name}_d_bins_scores.csv"
        bad_i_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_i_csv_filename)
        bad_d_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_d_csv_filename)
        
        print(f"Running inference on {subdir_name} with separate I and D models...")
        
        # Create temporary ID files for I and D bins
        temp_i_file, bad_i_bins = create_bin_type_id_file(subdir_path, "I")
        temp_d_file, bad_d_bins = create_bin_type_id_file(subdir_path, "D")
        
        print(f"Found {bad_i_bins} I bins and {bad_d_bins} D bins in {subdir_name}")
        
        try:
            # Run inference on I bins if any exist
            if bad_i_bins > 0:
                print(f"Processing {bad_i_bins} bad I bins with I model...")
                bad_i_inference_params = IFCBInferenceParams(
                    data_dir=subdir_path,
                    output_dir=ifcb_full_evaluation_params.output_dir,
                    model_path=ifcb_full_evaluation_params.i_model_path,
                    id_file=temp_i_file,
                    n_jobs=ifcb_full_evaluation_params.n_jobs,
                    aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                    chunk_size=ifcb_full_evaluation_params.chunk_size,
                    output_filename=bad_i_csv_filename
                )
                run_ifcb_flow_metric_inference(bad_i_inference_params, ifcb_image)
            
            # Run inference on D bins if any exist
            if bad_d_bins > 0:
                print(f"Processing {bad_d_bins} bad D bins with D model...")
                bad_d_inference_params = IFCBInferenceParams(
                    data_dir=subdir_path,
                    output_dir=ifcb_full_evaluation_params.output_dir,
                    model_path=ifcb_full_evaluation_params.d_model_path,
                    id_file=temp_d_file,
                    n_jobs=ifcb_full_evaluation_params.n_jobs,
                    aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                    chunk_size=ifcb_full_evaluation_params.chunk_size,
                    output_filename=bad_d_csv_filename
                )
                run_ifcb_flow_metric_inference(bad_d_inference_params, ifcb_image)
                
        finally:
            # Clean up temporary files
            if temp_i_file and os.path.exists(temp_i_file):
                os.unlink(temp_i_file)
            if temp_d_file and os.path.exists(temp_d_file):
                os.unlink(temp_d_file)
        
        print(f"{subdir_name} inference completed: {bad_i_bins} I bins, {bad_d_bins} D bins")
        
        # Create evaluation plots: separate I bins, D bins, and merged plots
        normal_data_clean = ifcb_full_evaluation_params.normal_data_name.replace(' ', '_').lower()
        
        # 1. I bins only evaluation (if both datasets have I bins)
        if bad_i_bins > 0 and normal_i_bins > 0:
            i_plot_filename = f"evaluation_{subdir_name}_vs_{normal_data_clean}_I_bins.png"
            i_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (I bins): {subdir_name} vs {ifcb_full_evaluation_params.normal_data_name}"
            
            i_evaluation_params = IFCBEvaluationParams(
                csv1_path=bad_i_csv_path,
                csv2_path=normal_i_csv_path,
                output_dir=ifcb_full_evaluation_params.output_dir,
                output_filename=i_plot_filename,
                title=i_plot_title,
                name1=f"{subdir_name} (I bins)",
                name2=f"{ifcb_full_evaluation_params.normal_data_name} (I bins)"
            )
            
            run_ifcb_flow_metric_evaluation(i_evaluation_params, ifcb_image)
            print(f"Created I bins evaluation plot: {i_plot_filename}")
        
        # 2. D bins only evaluation (if both datasets have D bins)
        if bad_d_bins > 0 and normal_d_bins > 0:
            d_plot_filename = f"evaluation_{subdir_name}_vs_{normal_data_clean}_D_bins.png"
            d_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (D bins): {subdir_name} vs {ifcb_full_evaluation_params.normal_data_name}"
            
            d_evaluation_params = IFCBEvaluationParams(
                csv1_path=bad_d_csv_path,
                csv2_path=normal_d_csv_path,
                output_dir=ifcb_full_evaluation_params.output_dir,
                output_filename=d_plot_filename,
                title=d_plot_title,
                name1=f"{subdir_name} (D bins)",
                name2=f"{ifcb_full_evaluation_params.normal_data_name} (D bins)"
            )
            
            run_ifcb_flow_metric_evaluation(d_evaluation_params, ifcb_image)
            print(f"Created D bins evaluation plot: {d_plot_filename}")
        
        # 3. Merged evaluation (combining I and D bins)
        if (bad_i_bins > 0 or bad_d_bins > 0) and (normal_i_bins > 0 or normal_d_bins > 0):
            # Merge bad data CSV files
            bad_merged_filename = f"bad_data_{subdir_name}_merged_scores.csv"
            bad_merged_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_merged_filename)
            
            merge_csv_files(
                csv_files=[bad_i_csv_path, bad_d_csv_path],
                output_path=bad_merged_path
            )
            
            # Merge normal data CSV files (only once per flow run)
            normal_merged_filename = "normal_data_merged_scores.csv"
            normal_merged_path = os.path.join(ifcb_full_evaluation_params.output_dir, normal_merged_filename)
            
            # Only create merged normal file if it doesn't exist yet
            if not os.path.exists(normal_merged_path):
                merge_csv_files(
                    csv_files=[normal_i_csv_path, normal_d_csv_path],
                    output_path=normal_merged_path
                )
            
            # Create merged evaluation plot
            merged_plot_filename = f"evaluation_{subdir_name}_vs_{normal_data_clean}_merged.png"
            merged_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (All bins): {subdir_name} vs {ifcb_full_evaluation_params.normal_data_name}"
            
            merged_evaluation_params = IFCBEvaluationParams(
                csv1_path=bad_merged_path,
                csv2_path=normal_merged_path,
                output_dir=ifcb_full_evaluation_params.output_dir,
                output_filename=merged_plot_filename,
                title=merged_plot_title,
                name1=f"{subdir_name} (All bins)",
                name2=f"{ifcb_full_evaluation_params.normal_data_name} (All bins)"
            )
            
            run_ifcb_flow_metric_evaluation(merged_evaluation_params, ifcb_image)
            print(f"Created merged evaluation plot: {merged_plot_filename}")
        
        print(f"Completed evaluation for {subdir_name}")
    
    print(f"Full evaluation completed for {len(bad_subdirs)} bad data categories")


if __name__ == "__main__":
    ifcb_full_evaluation_flow.serve(
        name="ifcb-flow-metric-full-evaluation",
        tags=["evaluation", "ifcb", "anomaly-detection", "comparison"],
    )
