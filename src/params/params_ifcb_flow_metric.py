from pydantic import BaseModel, Field
from typing import Optional, Union


# Feature selection configuration for IFCB flow metric training
class IFCBFeatureConfig(BaseModel):
    # Spatial Statistics Features
    use_mean_x: bool = Field(True, description="Use mean x-coordinate feature")
    use_mean_y: bool = Field(True, description="Use mean y-coordinate feature")
    use_std_x: bool = Field(True, description="Use standard deviation of x-coordinates feature")
    use_std_y: bool = Field(True, description="Use standard deviation of y-coordinates feature")
    use_median_x: bool = Field(True, description="Use median x-coordinate feature")
    use_median_y: bool = Field(True, description="Use median y-coordinate feature")
    use_iqr_x: bool = Field(True, description="Use interquartile range of x-coordinates feature")
    use_iqr_y: bool = Field(True, description="Use interquartile range of y-coordinates feature")
    
    # Distribution Shape Features
    use_ratio_spread: bool = Field(True, description="Use ratio of y-direction IQR to x-direction IQR feature")
    use_core_fraction: bool = Field(True, description="Use fraction of points within IQR box feature")
    
    # Clipping Detection Features
    use_duplicate_fraction: bool = Field(True, description="Use fraction of points sharing same y-value feature")
    use_max_duplicate_fraction: bool = Field(True, description="Use largest fraction of points with identical y-values feature")
    
    # Histogram Uniformity Features
    use_cv_x: bool = Field(True, description="Use coefficient of variation for x-coordinate histogram feature")
    use_cv_y: bool = Field(True, description="Use coefficient of variation for y-coordinate histogram feature")
    
    # Statistical Moments Features
    use_skew_x: bool = Field(True, description="Use skewness of x-coordinate distribution feature")
    use_skew_y: bool = Field(True, description="Use skewness of y-coordinate distribution feature")
    use_kurt_x: bool = Field(True, description="Use kurtosis of x-coordinate distribution feature")
    use_kurt_y: bool = Field(True, description="Use kurtosis of y-coordinate distribution feature")
    
    # PCA Orientation Features
    use_angle: bool = Field(True, description="Use principal component angle feature")
    use_eigen_ratio: bool = Field(True, description="Use ratio of first to second eigenvalue feature")
    
    # Edge Features
    use_left_edge_fraction: bool = Field(True, description="Use fraction of points near left frame edge feature")
    use_right_edge_fraction: bool = Field(True, description="Use fraction of points near right frame edge feature")
    use_top_edge_fraction: bool = Field(True, description="Use fraction of points near top frame edge feature")
    use_bottom_edge_fraction: bool = Field(True, description="Use fraction of points near bottom frame edge feature")
    use_total_edge_fraction: bool = Field(True, description="Use total fraction of points near any frame edge feature")
    
    # Temporal Features
    use_second_t_value: bool = Field(True, description="Use second timestamp value feature")
    use_t_var: bool = Field(True, description="Use variance of timestamp differences feature")

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
    
    # Feature selection configuration
    feature_config: IFCBFeatureConfig = Field(default_factory=IFCBFeatureConfig, description="Feature selection configuration")