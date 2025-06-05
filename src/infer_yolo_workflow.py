from prefect import flow

from params_amplify import YOLOInferenceParams, YOLOVisualizationParams
from tasks.run_yolo_inference import run_yolo_inference
from tasks.pull_images import pull_images

@flow(log_prints=True)
def yolo_infer(yolo_inference_params: YOLOInferenceParams, yolo_visualization_params: YOLOVisualizationParams):
    """Flow: Run YOLO using the given parameters."""
    image = 'ghcr.io/whoigit/amplify-prefect/amplify-ultralytics:latest'
    pull_images([image])
    run_yolo_inference(yolo_inference_params, yolo_visualization_params, image)

# Deploy the flow
if __name__ == "__main__":
    yolo_infer.serve(name="yolo-inference-local")
