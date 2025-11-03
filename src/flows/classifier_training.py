import os
from pathlib import Path
import shutil
from typing import Literal, List

from prefect import flow

from dojo.schemas import TrainingRunConfig

from src.tasks.run_containerized_classifier_training import run_container, VolumeMapping


@flow(log_prints=True)
def run_dojo_train_multiclass(output_dir: str, input_volumes: List[VolumeMapping], training_run_config: TrainingRunConfig, device_ids:List[str]=['all']):
    """Flow: Run Image Classifier Dojo using the given parameters."""

    run_container(output_dir, input_volumes, ['TRAIN','MULTICLASS'], training_run_config, device_ids)

# Deploy the flow
if __name__ == "__main__":
    run_dojo_train_multiclass.serve(name="image-classifier-dojo_train-multiclass")
