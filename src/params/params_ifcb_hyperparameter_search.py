from pydantic import BaseModel, Field
from typing import Optional, List, Union


class HyperparameterRange(BaseModel):
    """Defines a range or set of values for a hyperparameter."""
    values: Optional[List[Union[float, int]]] = Field(None, description="Specific values to try")
    min_val: Optional[float] = Field(None, description="Minimum value for range")
    max_val: Optional[float] = Field(None, description="Maximum value for range")
    steps: Optional[int] = Field(None, description="Number of steps in range (for min_val/max_val)")


class IFCBHyperparameterSearchParams(BaseModel):
    """Parameters for IFCB hyperparameter search."""
    
    # Base IFCB training parameters (fixed across all runs)
    data_dir: str = Field(..., description="Directory containing IFCB point cloud data")
    base_output_dir: str = Field(..., description="Base directory where all hyperparameter search results will be saved")
    id_file: Optional[str] = Field(None, description="File containing list of IDs to load (one PID per line)")
    n_jobs: int = Field(-1, description="Number of parallel jobs for load/extraction phase (-1 uses all CPUs)")
    model_filename: str = Field("classifier.pkl", description="Filename for the trained model")
    
    # Hyperparameter search ranges
    contamination_range: Optional[HyperparameterRange] = Field(
        None, 
        description="Range/values for contamination parameter (expected fraction of anomalous distributions)"
    )
    
    # Fixed parameters (not searched)
    aspect_ratio: float = Field(1.36, description="Camera frame aspect ratio (fixed for IFCB data)")
    chunk_size: int = Field(100, description="Number of PIDs to process in each chunk (fixed for efficiency)")
    
    # Default values for parameters not being searched
    default_contamination: float = Field(0.1, description="Default contamination if not searching")
