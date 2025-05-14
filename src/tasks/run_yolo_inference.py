from prefect import task
import subprocess

from prefect import get_run_logger

from prov import on_task_complete
from params_amplify import YOLOInferenceParams, YOLOVisualizationParams


@task(on_completion=[on_task_complete], log_prints=True)
def run_yolo_inference(yolo_inference_params: YOLOInferenceParams, yolo_visualization_params: YOLOVisualizationParams):
    """
    Run YOLO in a Podman container.
    """

    command = (
        f"podman run "
        f"-it --rm --gpus all --ipc host "
        f"-v {yolo_inference_params.data_dir}:/data "
        f"-v {yolo_inference_params.output_dir}:/output "
        f"-v {yolo_inference_params.model_weights_path}:/input/weights.pt "
        f"localhost/ultralytics:latest "
        f"/ultralytics/yolo_inference.sh "
        f"{yolo_inference_params.device} "
        f"{yolo_inference_params.agnostic_nms} "
        f"{yolo_inference_params.iou} "
        f"{yolo_inference_params.conf} "
        f"{yolo_inference_params.imgsz} "
        f"{yolo_inference_params.batch } "
        f"{yolo_inference_params.half} "
        f"{yolo_inference_params.max_det} "
        f"{yolo_inference_params.vid_stride} "
        f"{yolo_inference_params.stream_buffer} "
        f"{yolo_inference_params.visualize} "
        f"{yolo_inference_params.augment} "
        f"{yolo_inference_params.classes} "
        f"{yolo_inference_params.retina_masks} "
        f"{yolo_inference_params.embed} "
        f"{yolo_inference_params.name} "
        f"{yolo_inference_params.verbose} "
        f"{yolo_visualization_params.show} "
        f"{yolo_visualization_params.save} "
        f"{yolo_visualization_params.save_frames} "
        f"{yolo_visualization_params.save_txt} "
        f"{yolo_visualization_params.save_conf} "
        f"{yolo_visualization_params.save_crop} "
        f"{yolo_visualization_params.show_labels} "
        f"{yolo_visualization_params.show_conf} "
        f"{yolo_visualization_params.show_boxes}"
        )
    
    logger = get_run_logger()

    process = subprocess.Popen(
        command, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    for line in process.stdout:
        logger.info(line)

