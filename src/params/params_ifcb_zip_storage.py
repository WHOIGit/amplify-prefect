from pydantic import BaseModel, Field
from typing import Optional


class IFCBZipStorageParams(BaseModel):
    data_dir: str = Field(
        ...,
        description="Path to IFCB data directory"
    )

    storage_yaml: str = Field(
        ...,
        description="Path to YAML file defining the object store configuration"
    )

    env_file: str = Field(
        ...,
        description="Path to .env file containing environment variables for storage configuration"
    )
