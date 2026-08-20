from prefect import flow, get_run_logger

from src.params.params_onnx import ONNXInferenceParams
from src.tasks.pull_images import pull_images
from src.tasks.run_onnx_inference import run_onnx_inference
from src.utils.output_dir_utils import create_output_dir

DEFAULT_ONNX_IMAGE = "ghcr.io/whoigit/ifcb-inference:latest"
EMBEDDINGS_ONNX_IMAGE = "ghcr.io/whoigit/ifcb-inference-embeddings:latest"
OPTIONAL_OUTPUT_SUFFIXES = (".h5", ".parquet")


def _score_output_needs_optional_deps(onnx_inference_params: ONNXInferenceParams) -> bool:
    if onnx_inference_params.embeddings_only or onnx_inference_params.outfile is None:
        return False
    return onnx_inference_params.outfile.lower().endswith(OPTIONAL_OUTPUT_SUFFIXES)


def _select_onnx_image(onnx_inference_params: ONNXInferenceParams) -> str:
    if (
        onnx_inference_params.embeddings
        or onnx_inference_params.embeddings_only
        or _score_output_needs_optional_deps(onnx_inference_params)
    ):
        return EMBEDDINGS_ONNX_IMAGE
    return DEFAULT_ONNX_IMAGE

@flow(log_prints=True)
def onnx_infer(onnx_inference_params: ONNXInferenceParams):
    """Flow: Run ONNX inference using the given parameters."""
    logger = get_run_logger()

    # Create the output directory before the container starts. This container
    # runs as root, so if the bind-mount source is missing Docker creates it
    # root:root 0755 with no setgid -- which the image's umask cannot fix.
    create_output_dir(onnx_inference_params.output_dir, logger)
    logger.info(f"Output directory: {onnx_inference_params.output_dir}")

    image = _select_onnx_image(onnx_inference_params)
    pull_images([image])
    run_onnx_inference(onnx_inference_params, image)

# Deploy the flow
if __name__ == "__main__":
    onnx_infer.serve(name="onnx-inference-local")
