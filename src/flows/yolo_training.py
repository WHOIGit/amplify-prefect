import os
from pathlib import Path
import shutil

from prefect import flow

from src.params.params_amplify import YOLOTrainParams
from src.tasks.run_containerized_yolo import run_containerized_yolo


@flow(log_prints=True)
def run_yolo(yolo_params: YOLOTrainParams):
    """Flow: Run YOLO using the given parameters."""

    run_containerized_yolo(yolo_params.data_dir, yolo_params.output_dir, yolo_params.model_name, yolo_params.epochs, yolo_params.gpus)

# Deploy the flow
if __name__ == "__main__":
    run_yolo.serve(name="yolo-training")
