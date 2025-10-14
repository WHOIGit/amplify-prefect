from pydantic import BaseModel, Field
from typing import Optional, List


class FeatureValidationParams(BaseModel):
    """Parameters for IFCB feature validation."""

    # Docker image
    validation_image: str = Field(..., description="Docker image for validation")

    # VastDB connection
    vastdb_url: str = Field(..., description="VastDB endpoint URL")

    # Predicted features table
    pred_bucket: str = Field(..., description="VastDB bucket name for predicted features")
    pred_schema: str = Field(..., description="VastDB schema name for predicted features")
    pred_table: str = Field(..., description="VastDB table name for predicted features")

    # Ground truth features table
    gt_bucket: str = Field(..., description="VastDB bucket name for ground truth features")
    gt_schema: str = Field(..., description="VastDB schema name for ground truth features")
    gt_table: str = Field(..., description="VastDB table name for ground truth features")

    # Output configuration
    output_directory: str = Field(..., description="Path to the directory to save validation results")
    output_filename: str = Field("validation_results.csv", description="CSV filename for detailed metrics")
    summary_filename: str = Field("validation_summary.json", description="JSON filename for summary statistics")

    # Column name mappings
    pred_sample_col: str = Field("sample_id", description="Sample ID column name in predicted table")
    gt_sample_col: str = Field("ifcb_bin", description="Sample ID column name in ground truth table")
    pred_roi_col: str = Field("roi_number", description="ROI number column name in predicted table")
    gt_roi_col: str = Field("roi_number", description="ROI number column name in ground truth table")

    # Optional filters
    sample_ids: Optional[List[str]] = Field(None, description="Optional list of specific sample IDs to validate")

    # Blob comparison configuration
    enable_blob_comparison: bool = Field(False, description="Enable pixel-by-pixel blob comparison")
    blob_pred_bucket: Optional[str] = Field(None, description="S3 bucket containing predicted blobs")
    blob_gt_bucket: Optional[str] = Field(None, description="S3 bucket containing ground truth blobs")
    blob_s3_url: Optional[str] = Field(None, description="S3 endpoint URL for blobs")
    blob_pred_prefix: str = Field("ifcb-blobs-slim-features/", description="S3 prefix for predicted blobs")
    blob_gt_prefix: str = Field("ifcb-blobs-gt/", description="S3 prefix for ground truth blobs")
    blob_top_n_worst: int = Field(10, description="Number of worst blob matches to visualize")
