from pydantic import BaseModel, Field, field_validator
import os


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

    num_workers: int = Field(
        16,
        description="Number of parallel workers for processing bins (capped at CPU count)"
    )

    @field_validator('num_workers')
    @classmethod
    def cap_workers_at_cpu_count(cls, v):
        max_workers = os.cpu_count() or 1
        if v > max_workers:
            return max_workers
        if v < 1:
            return 1
        return v
