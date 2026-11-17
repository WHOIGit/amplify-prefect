from prefect import task
import docker

from prefect import get_run_logger

from src.params.params_amplify import YOLOInferenceParams, YOLOVisualizationParams
from src.tasks.gpu import allocate_gpus, container_devices, run_container


def _build_command_args(
    yolo_inference_params: YOLOInferenceParams,
    yolo_visualization_params: YOLOVisualizationParams,
    device: str | None = None,
) -> list[str]:
    """``device`` overrides yolo_inference_params.device, e.g. with the in-container GPU indices."""
    command_args = [
        "python3",
        "/ultralytics/yolo_inference.py",
        str(yolo_inference_params.device if device is None else device),
        str(yolo_inference_params.agnostic_nms),
        str(yolo_inference_params.iou),
        str(yolo_inference_params.conf),
        str(yolo_inference_params.imgsz),
        str(yolo_inference_params.batch),
        str(yolo_inference_params.half),
        str(yolo_inference_params.max_det),
        str(yolo_inference_params.vid_stride),
        str(yolo_inference_params.stream_buffer),
        str(yolo_inference_params.visualize),
        str(yolo_inference_params.augment),
        str(yolo_inference_params.classes),
        str(yolo_inference_params.retina_masks),
        str(yolo_inference_params.embed),
        str(yolo_inference_params.name),
        str(yolo_inference_params.verbose),
        str(yolo_visualization_params.show),
        str(yolo_visualization_params.save),
        str(yolo_visualization_params.save_frames),
        str(yolo_visualization_params.save_txt),
        str(yolo_visualization_params.save_conf),
        str(yolo_visualization_params.save_crop),
        str(yolo_visualization_params.show_labels),
        str(yolo_visualization_params.show_conf),
        str(yolo_visualization_params.show_boxes),
    ]

    if yolo_inference_params.ext:
        command_args.extend(["--ext", yolo_inference_params.ext])
    if yolo_inference_params.max_files is not None:
        command_args.extend(["--max-files", str(yolo_inference_params.max_files)])
    if yolo_inference_params.skip_validation:
        command_args.append("--skip-validation")
    if yolo_inference_params.validation_workers is not None:
        command_args.extend(["--validation-workers", str(yolo_inference_params.validation_workers)])
    if yolo_inference_params.files_per_call is not None:
        command_args.extend(["--files-per-call", str(yolo_inference_params.files_per_call)])
    if yolo_inference_params.convert_workers is not None:
        command_args.extend(["--convert-workers", str(yolo_inference_params.convert_workers)])

    return command_args


@task(log_prints=True)
def run_yolo_inference(yolo_inference_params: YOLOInferenceParams, yolo_visualization_params: YOLOVisualizationParams, yolo_image: str):
    """
    Run YOLO in a Docker container.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        yolo_inference_params.data_dir: {'bind': '/data', 'mode': 'rw'},
        yolo_inference_params.output_dir: {'bind': '/output', 'mode': 'rw'},
        yolo_inference_params.model_weights_path: {'bind': '/input/weights.pt', 'mode': 'ro'}
    }
    
    with allocate_gpus(yolo_inference_params.device, logger) as gpus:
        # The container only sees the allocated GPUs, renumbered from 0.
        device = container_devices(gpus) if gpus else "cpu"
        command_args = _build_command_args(yolo_inference_params, yolo_visualization_params, device=device)
        logger.info(f'Running container with command: {" ".join(command_args)}')
        run_container(
            client,
            yolo_image,
            command_args,
            logger,
            gpus=gpus,
            volumes=volumes,
            ipc_mode="host",
        )
