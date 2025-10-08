from pydantic import BaseModel, Field
from typing import Optional, List


class ExtractSlimFeaturesParams(BaseModel):
    # Docker image
    extract_features_image: str = Field(..., description="Docker image for feature extraction")

    data_directory: str = Field(..., description="Path to the directory containing IFCB data")
    output_directory: str = Field(..., description="Path to the directory to save output (for local storage modes)")
    bins: Optional[List[str]] = Field(None, description="List of bin names to process (e.g., 'D20240423T115846_IFCB127'). If None, all bins are processed")

    # Blob storage options
    blob_storage_mode: str = Field("s3", description="Storage mode for blobs: 'local' or 's3'")
    s3_bucket: Optional[str] = Field(None, description="S3 bucket name (required when blob_storage_mode=s3)")
    s3_url: Optional[str] = Field(None, description="S3 endpoint URL (required when blob_storage_mode=s3)")
    s3_prefix: str = Field("ifcb-blobs-slim-features/", description="S3 prefix for blob storage")

    # Feature storage options
    feature_storage_mode: str = Field("vastdb", description="Storage mode for features: 'local' or 'vastdb'")
    vastdb_bucket: Optional[str] = Field(None, description="VastDB bucket name (required when feature_storage_mode=vastdb)")
    vastdb_schema: Optional[str] = Field(None, description="VastDB schema name (required when feature_storage_mode=vastdb)")
    vastdb_table: Optional[str] = Field(None, description="VastDB table name (required when feature_storage_mode=vastdb)")
    vastdb_url: Optional[str] = Field(None, description="VastDB endpoint URL (defaults to s3_url if not provided)")

    # GPU batch processing options
    batch_processing: bool = Field(False, description="Enable GPU-accelerated batch processing for phase congruency")
    min_batch_size: int = Field(4, description="Minimum number of ROIs needed to form a batch")
    max_batch_size: int = Field(64, description="Maximum batch size for GPU memory management")
    gpu_device: Optional[int] = Field(None, description="GPU device index to use (e.g., 0, 1, 2). If None, uses default device")
