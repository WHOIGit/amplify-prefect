from pydantic import BaseModel, Field
from typing import Optional, Union

# Parameters relevant to IFCB flow metric training
class IFCBTrainingParams(BaseModel):
    data_dir: str = Field(..., description="Directory containing IFCB point cloud data")
    output_dir: str = Field(..., description="Directory where trained model will be saved")
    id_file: Optional[str] = Field(None, description="File containing list of IDs to load (one PID per line)")
    n_jobs: int = Field(-1, description="Number of parallel jobs for load/extraction phase (-1 uses all CPUs)")
    contamination: float = Field(0.1, description="Expected fraction of anomalous distributions")
    aspect_ratio: float = Field(1.36, description="Camera frame aspect ratio (width/height)")
    chunk_size: int = Field(100, description="Number of PIDs to process in each chunk")
    model_filename: str = Field("classifier.pkl", description="Filename for the trained model")
    max_samples: Union[int, float, str] = Field("auto", description="Number of samples to draw from X to train each base estimator")
    max_features: Union[int, float] = Field(1.0, description="Number of features to draw from X to train each base estimator")