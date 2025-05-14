from prefect import flow

from params_amplify import YOLOInferenceParams, YOLOVisualizationParams
from tasks.run_yolo_inference import run_yolo_inference


@flow(log_prints=True)
def yolo_infer(yolo_inference_params: YOLOInferenceParams, yolo_visualization_params: YOLOVisualizationParams):
    """Flow: Run YOLO using the given parameters."""

    run_yolo_inference(yolo_inference_params, yolo_visualization_params)

# Deploy the flow
if __name__ == "__main__":
    yolo_infer.serve(name="yolo-inference")
