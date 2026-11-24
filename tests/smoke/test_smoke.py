"""Each flow, run for real on a small input. See conftest.py for the sample data."""

import shutil

import pandas as pd
import pytest

from src.flows.classifier_training import run_dojo_train_multiclass
from src.flows.extract_slim_features import extract_slim_features_flow
from src.flows.feature_validation import feature_validation_flow
from src.flows.ifcb_full_evaluation import ifcb_full_evaluation_flow
from src.flows.ifcb_hyperparameter_search import ifcb_hyperparameter_search_flow
from src.flows.ifcb_inference import ifcb_inference_flow
from src.flows.ifcb_training import ifcb_training_flow
from src.flows.ifcb_zip_storage import ifcb_zip_storage
from src.flows.onnx_inference import onnx_infer
from src.flows.stingray_frame_timestamps import stingray_frame_timestamps
from src.flows.yolo_inference import yolo_infer
from src.flows.yolo_training import run_yolo
from src.params.params_amplify import YOLOInferenceParams, YOLOTrainParams, YOLOVisualizationParams
from src.params.params_extract_slim_features import ExtractSlimFeaturesParams
from src.params.params_feature_validation import FeatureValidationParams
from src.params.params_ifcb_flow_metric import IFCBFullEvaluationParams, IFCBInferenceParams, IFCBTrainingParams
from src.params.params_ifcb_hyperparameter_search import IFCBHyperparameterSearchParams
from src.params.params_ifcb_zip_storage import IFCBZipStorageParams
from src.params.params_onnx import ONNXInferenceParams
from src.params.params_stingray_image_analysis import FrameTimestampParams, StingrayCruiseParams

# Every flow that uses a GPU takes one idle GPU.
ONE_GPU = "auto:1"


def bins_in(directory, pattern="*.adc"):
    return sorted(p.stem for p in directory.rglob(pattern))


def files_in(directory, pattern):
    return sorted(directory.rglob(pattern))


def run(flow, *args):
    state = flow(*args, return_state=True)
    if state.is_failed() and "GpuUnavailableError" in (state.message or ""):
        # Every GPU stayed busy with real work; that says nothing about the flow.
        pytest.skip(state.message)
    assert state.is_completed(), state.message
    return state


def test_onnx_inference(data, tmp_path):
    inputs = data / "ifcb/normal/2026"
    params = ONNXInferenceParams(model=str(data / "onnx/classifier.onnx"), classes=str(data / "onnx/classes.list"),
                                 input_dir=str(inputs), output_dir=str(tmp_path / "out"),
                                 cuda_visible_devices=ONE_GPU)
    run(onnx_infer, params)
    outputs = files_in(tmp_path / "out", "*.csv")
    assert sorted(p.stem for p in outputs) == bins_in(inputs)
    assert len(pd.read_csv(outputs[0])) > 0


def test_onnx_embeddings(data, tmp_path):
    inputs = data / "ifcb/normal/2026"
    params = ONNXInferenceParams(model=str(data / "onnx/embeddings.onnx"), classes=str(data / "onnx/classes.list"),
                                 input_dir=str(inputs), output_dir=str(tmp_path / "out"),
                                 embeddings=True, embeddings_only=True, batch=64, cuda_visible_devices=ONE_GPU)
    run(onnx_infer, params)
    outputs = files_in(tmp_path / "out", "*.emb.parquet")
    assert sorted(p.name.split(".")[0] for p in outputs) == bins_in(inputs)
    assert all(p.stat().st_size > 0 for p in outputs)


def test_yolo_inference(data, copy_of, tmp_path):
    images = copy_of(data / "yolo/images")
    params = YOLOInferenceParams(data_dir=str(images), output_dir=str(tmp_path / "out"),
                                 model_weights_path=str(data / "yolo/best.pt"), ext=".tiff", imgsz=640, batch=4,
                                 device=ONE_GPU)
    run(yolo_infer, params, YOLOVisualizationParams())
    assert any((tmp_path / "out").rglob("*"))


def test_yolo_training(data, copy_of, tmp_path):
    dataset = copy_of(data / "yolo/train")
    # Start from our own weights, so nothing is downloaded.
    shutil.copy(data / "yolo/best.pt", dataset / "start.pt")
    params = YOLOTrainParams(data_dir=str(dataset), output_dir=str(tmp_path / "out"), model_name="/data/start",
                             epochs=1, imgsz=320, batch=4, gpus=ONE_GPU)
    run(run_yolo, params)
    assert files_in(tmp_path / "out", "best.pt")


def test_ifcb_training(data, tmp_path):
    params = IFCBTrainingParams(data_dir=str(data / "ifcb/normal"), output_dir=str(tmp_path / "out"), bin_type="D")
    run(ifcb_training_flow, params)
    assert (tmp_path / "out/classifier.pkl").exists()


def test_ifcb_inference(data, tmp_path):
    inputs = data / "ifcb/normal/2026"
    params = IFCBInferenceParams(data_dir=str(inputs), output_dir=str(tmp_path / "out"),
                                 model_path=str(data / "flow-metric/d.pkl"), output_filename="scores.csv")
    run(ifcb_inference_flow, params)
    scores = pd.read_csv(tmp_path / "out/scores.csv")
    assert len(scores) == len(bins_in(inputs))


def test_ifcb_hyperparameter_search(data, tmp_path):
    params = IFCBHyperparameterSearchParams(data_dir=str(data / "ifcb/normal"), base_output_dir=str(tmp_path / "search"),
                                            bin_type="D", contamination_range={"values": [0.05, 0.1]})
    results = run(ifcb_hyperparameter_search_flow, params).result()
    assert [r["status"] for r in results] == ["success", "success"], results


def test_ifcb_full_evaluation(data, tmp_path):
    params = IFCBFullEvaluationParams(
        bad_i_data_dir=str(data / "ifcb/bad/i"), bad_d_data_dir=str(data / "ifcb/bad/d"),
        normal_data_dir=str(data / "ifcb/normal"), normal_data_name="Normal",
        i_model_path=str(data / "flow-metric/i.pkl"), d_model_path=str(data / "flow-metric/d.pkl"),
        output_dir=str(tmp_path / "out"),
    )
    run(ifcb_full_evaluation_flow, params)
    out = tmp_path / "out"
    assert len(pd.read_csv(out / "bad_merged_scores.csv")) > 0
    assert len(pd.read_csv(out / "normal_data_merged_scores.csv")) == len(bins_in(data / "ifcb/normal"))
    assert files_in(out, "*.png")


def test_extract_slim_features(data, tmp_path):
    inputs = data / "ifcb/normal/2026"
    pid = bins_in(inputs)[0]
    params = ExtractSlimFeaturesParams(extract_features_source="main", data_directory=str(inputs),
                                       output_directory=str(tmp_path / "out"), bins=[pid], workers=2,
                                       blob_storage_mode="local", feature_storage_mode="local")
    run(extract_slim_features_flow, params)
    outputs = list((tmp_path / "out").rglob(f"*{pid}*"))
    assert outputs, f"no output for {pid}"


def test_ifcb_zip_storage(data, external, tmp_path):
    settings = external.get("zip_storage")
    if not settings:
        pytest.skip("no zip_storage inputs in smoke.json")
    from prefect_aws import AwsCredentials

    credentials = AwsCredentials.load("vasts3-creds")
    env = tmp_path / "storage.env"
    env.write_text(f"AWS_ACCESS_KEY_ID={credentials.aws_access_key_id}\n"
                   f"AWS_SECRET_ACCESS_KEY={credentials.aws_secret_access_key.get_secret_value()}\n")
    storage = tmp_path / "storage.yaml"
    storage.write_text(f"""stores:
  bucket:
    type: AsyncBucketStore
    config:
      endpoint_url: {settings["endpoint_url"]}
      s3_access_key: ${{AWS_ACCESS_KEY_ID}}
      s3_secret_key: ${{AWS_SECRET_ACCESS_KEY}}
      bucket_name: {settings["bucket"]}
  prefixed:
    type: AsyncPrefixStore
    config:
      prefix: {settings["prefix"]}
    base: bucket
main: prefixed
""")
    params = IFCBZipStorageParams(data_dir=str(data / "ifcb/normal"), storage_yaml=str(storage), env_file=str(env),
                                  num_workers=2)
    run(ifcb_zip_storage, params)


def test_feature_validation(external, tmp_path):
    settings = external.get("feature_validation")
    if not settings:
        pytest.skip("no feature_validation inputs in smoke.json")
    params = FeatureValidationParams(output_directory=str(tmp_path / "out"), **settings)
    run(feature_validation_flow, params)
    assert len(pd.read_csv(tmp_path / "out/validation_results.csv")) > 0


def test_stingray_frame_timestamps(external, tmp_path):
    settings = external.get("stingray")
    if not settings:
        pytest.skip("no stingray inputs in smoke.json")
    cruise = StingrayCruiseParams(cruise=settings["cruise"], cruise_date=settings["cruise_date"],
                                  camera_stream=settings["camera_stream"], stingray_data_root=str(tmp_path))
    run(stingray_frame_timestamps, cruise,
        FrameTimestampParams(video_data_root=settings["video_data_root"], file_limit=1, max_workers=1))
    assert files_in(tmp_path / "media_list", "*")


def test_classifier_training(external, tmp_path):
    settings = external.get("dojo")
    if not settings:
        pytest.skip("no dojo inputs in smoke.json")
    volumes = [
        {"host_path": settings["list_dir"], "container_path": "/listfiles", "mode": "ro"},
        {"host_path": settings["data_dir"], "container_path": settings["data_dir"], "mode": "ro"},
    ]
    config = {
        "logger": {"repo": "/workspace/experiments/aim", "experiment": "smoke"},
        "dataset_config": {"dataset": {"classlist": "/listfiles/class_labels.list",
                                       "trainlist": "/listfiles/train.list", "vallist": "/listfiles/val.list"}},
        "model": {"backbone": {"model_name": "efficientnet_b0", "pretrained_weights": None}, "head": {}},
        "training": {"batch_size": 8, "model_optims": {"loss_config": {"label_smoothing": 0.1},
                                                        "optimizer_config": {"lr": 0.001}}},
        # One training and one validation batch.
        "runtime": {"fast_dev_run": True, "num_workers": 2, "experiment": "smoke", "run_name": "smoke"},
    }
    run(run_dojo_train_multiclass, str(tmp_path / "out"), volumes, config, [ONE_GPU])
