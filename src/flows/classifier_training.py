import os
from pathlib import Path
import shutil

from prefect import flow

from image_classifier_dojo.schemas import TrainingRunConfig

from tasks.run_containerized_classifier_training import run_container


@flow(log_prints=True)
def run_dojo_train_multiclass(datasets_dir:str, experiments_dir:str, training_run_config: TrainingRunConfig):
    """Flow: Run Image Classifier Dojo using the given parameters."""

    run_container(datasets_dir, experiments_dir, ['TRAIN','MULTICLASS'], training_run_config)

# Deploy the flow
if __name__ == "__main__":
    run_dojo_train_multiclass.serve(name="image-classifier-dojo_train-multiclass")
