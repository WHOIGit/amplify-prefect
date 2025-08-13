from pydantic import BaseModel, Field
from typing import Optional, List


class ExtractSlimFeaturesParams(BaseModel):
    data_directory: str = Field(..., description="Path to the directory containing IFCB data")
    output_directory: str = Field(..., description="Path to the directory to save the output CSV file and blobs")
    bins: Optional[List[str]] = Field(None, description="List of bin names to process (e.g., 'D20240423T115846_IFCB127'). If None, all bins are processed")
