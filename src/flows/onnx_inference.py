from prefect import flow

from src.params.params_onnx import ONNXInferenceParams
from src.tasks.run_onnx_inference import run_onnx_inference
from src.tasks.pull_images import pull_images

@flow(log_prints=True)
def onnx_infer(onnx_inference_params: ONNXInferenceParams):
    """Flow: Run ONNX inference using the given parameters."""
    image = 'ghcr.io/whoigit/amplify-prefect/amplify-onnx:latest'
    pull_images([image])
    run_onnx_inference(onnx_inference_params, image)

# Deploy the flow
if __name__ == "__main__":
    onnx_infer.serve(name="onnx-inference-local")
