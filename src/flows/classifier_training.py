import os
from pathlib import Path
import shutil
from typing import Literal, List

from prefect import flow

from src.tasks.run_containerized_classifier_training import (
    ARTIFACTS_CREDENTIALS_BLOCK,
    DojoTrainingRunConfig,
    VolumeMapping,
    run_dojo_container,
)


@flow(log_prints=True)
def run_dojo_train_multiclass(output_dir: str, input_volumes: List[VolumeMapping], training_run_config: DojoTrainingRunConfig, device_ids:List[str]=['all'], artifacts_credentials_block: str = ARTIFACTS_CREDENTIALS_BLOCK):
    """Flow: Run Image Classifier Dojo using the given parameters.

    ``device_ids`` is a GPU spec (see src/tasks/gpu.py); ['all'] means every idle GPU.
    ``artifacts_credentials_block`` names the AWS Credentials block holding the keys
    for an S3 ``artifacts_location``; keys are never flow parameters.
    """

    run_dojo_container(output_dir, input_volumes, ['TRAIN','MULTICLASS'], training_run_config, device_ids,
                       artifacts_credentials_block)

# Deploy the flow
if __name__ == "__main__":
    run_dojo_train_multiclass.serve(name="image-classifier-dojo_train-multiclass")
