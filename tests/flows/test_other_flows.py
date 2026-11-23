import json
import os
from pathlib import Path

import pytest

from src.flows.extract_slim_features import extract_slim_features_flow
from src.flows.feature_validation import feature_validation_flow
from src.flows.stingray_frame_timestamps import stingray_frame_timestamps
from src.flows.stingray_image_abundance import stingray_image_abundance
from src.params.params_extract_slim_features import ExtractSlimFeaturesParams
from src.params.params_feature_validation import FeatureValidationParams
from src.params.params_stingray_image_analysis import (
    FrameTimestampParams,
    ImageAbundanceParams,
    StingrayCruiseParams,
)


def test_extract_slim_features_main_image(fake_docker, tmp_path):
    params = ExtractSlimFeaturesParams(extract_features_source="main", data_directory=str(tmp_path / "data"),
                                       output_directory=str(tmp_path / "out"), bins=["D20240101T000000_IFCB101"],
                                       workers=2)
    state = extract_slim_features_flow(params, return_state=True)

    assert state.is_completed(), state.message
    assert fake_docker.pulls == ["ghcr.io/whoigit/ifcb-features:latest"]
    call = fake_docker.call
    assert call.args == ["/app/data", "/app/output", "--workers", "2", "--bins", "D20240101T000000_IFCB101"]
    assert call.gpu_ids == []
    assert call.kwargs["user"] == f"{os.getuid()}:{os.getgid()}"
    assert os.path.isdir(params.output_directory)


def test_extract_slim_features_storage_image_on_gpu(fake_docker, tmp_path):
    params = ExtractSlimFeaturesParams(
        extract_features_source="storage", extract_features_image="registry.example.org/features:test",
        data_directory=str(tmp_path / "data"), output_directory=str(tmp_path / "out"),
        s3_bucket="blobs", s3_url="https://s3.example.org",
        vastdb_bucket="db", vastdb_schema="schema", vastdb_table="features",
        batch_processing=True,
    )
    state = extract_slim_features_flow(params, return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.image == "registry.example.org/features:test"
    assert call.arg("--s3-bucket") == "blobs"
    assert call.arg("--vastdb-url") == "https://s3.example.org"
    # One idle GPU, which the container sees as device 0.
    assert call.gpu_ids == ["0"]
    assert call.arg("--gpu-device") == "0"
    assert call.environment["AWS_ACCESS_KEY_ID"] == "test-key"
    assert call.environment["AWS_SECRET_ACCESS_KEY"] == "test-secret"


FEATURE_SUMMARY = {
    "num_rows_analyzed": 10, "num_samples_analyzed": 1, "samples_analyzed": ["D20240101T000000_IFCB101"],
    "total_features": 2, "mean_rmse": 0.1, "median_rmse": 0.1, "mean_mae": 0.1, "median_mae": 0.1,
    "mean_r2": 0.9, "median_r2": 0.9, "mean_pearson_r": 0.95, "median_pearson_r": 0.95,
    "features_with_high_correlation": 2, "features_with_low_r2": 0,
}
BLOB_SUMMARY = {"summary_stats": {
    "total_blobs": 4, "mean_iou": 0.9, "median_iou": 0.9, "std_iou": 0.05, "mean_dice": 0.95,
    "median_dice": 0.95, "mean_accuracy": 0.99, "perfect_matches": 1, "near_perfect_matches": 2, "poor_matches": 0,
}}


@pytest.fixture
def validation_params(tmp_path):
    def make(**overrides):
        values = dict(
            validation_image="registry.example.org/validation:test", vastdb_url="https://vast.example.org",
            pred_bucket="pb", pred_schema="ps", pred_table="pt", gt_bucket="gb", gt_schema="gs", gt_table="gt",
            output_directory=str(tmp_path / "out"),
        )
        values.update(overrides)
        return FeatureValidationParams(**values)

    return make


def test_feature_validation_with_blob_comparison(fake_docker, validation_params, tmp_path):
    @fake_docker.on_run
    def write_results(call):
        if call.args[0] == "validate_features.py":
            call.host_path(call.arg("--summary")).write_text(json.dumps(FEATURE_SUMMARY))
            call.host_path(call.arg("--output")).write_text(
                "feature,r2,rmse,mae,pearson_r\narea,0.95,0.1,0.1,0.97\nperimeter,0.85,0.2,0.1,0.92\n"
            )
        else:
            out = call.host_path(call.arg("--output-dir"))
            (out / "blob_comparison_results.json").write_text(json.dumps(BLOB_SUMMARY))
            (out / "blob_comparison_details.csv").write_text(
                "sample_id,roi_number,iou,dice,accuracy,diff_pixels\nD20240101T000000_IFCB101,1,0.9,0.95,0.99,3\n"
            )

    params = validation_params(enable_blob_comparison=True, blob_pred_bucket="pb", blob_gt_bucket="gb",
                               blob_s3_url="https://s3.example.org", sample_ids=["D20240101T000000_IFCB101"])
    state = feature_validation_flow(params, return_state=True)

    assert state.is_completed(), state.message
    assert [c.args[0] for c in fake_docker.calls] == ["validate_features.py", "compare_blobs.py"]
    validate, compare = fake_docker.calls
    assert validate.arg("--pred-table") == "pt"
    assert validate.arg("--sample-ids") == "D20240101T000000_IFCB101"
    assert validate.environment["AWS_VASTDB_URL"] == "https://vast.example.org"
    assert validate.environment["AWS_SECRET_ACCESS_KEY"] == "test-secret"
    assert compare.mount("/app/blob_comparison_output") == str(tmp_path / "out" / "blob_comparison")
    assert all(c.removed for c in fake_docker.containers_made)


def test_feature_validation_without_blob_comparison(fake_docker, validation_params):
    state = feature_validation_flow(validation_params(), return_state=True)
    assert state.is_completed(), state.message
    assert [c.args[0] for c in fake_docker.calls] == ["validate_features.py"]


@pytest.fixture
def cruise(tmp_path):
    (tmp_path / "stingray").mkdir()
    return StingrayCruiseParams(cruise="EN706", cruise_date="20231001", camera_stream="cam1",
                                stingray_data_root=str(tmp_path / "stingray"))


def test_stingray_frame_timestamps(fake_docker, cruise, tmp_path):
    (tmp_path / "videos").mkdir()
    seen = {}

    @fake_docker.on_run
    def read_config(call):
        seen["config"] = call.host_path("/run/cruise.conf.sh").read_text()

    state = stingray_frame_timestamps(cruise, FrameTimestampParams(video_data_root=str(tmp_path / "videos"),
                                                                   max_workers=3), return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.image == "ghcr.io/whoigit/stingray-image-analysis:latest"
    assert call.args == ["bash", "frame_timestamps.sh", "/run/cruise.conf.sh"]
    assert call.mount("/data/videos") == str(tmp_path / "videos")
    assert call.kwargs["nano_cpus"] == 4_000_000_000
    assert "CRUISE=EN706" in seen["config"]
    assert "TIMESTAMP_MAX_WORKERS=3" in seen["config"]
    assert not Path(call.mount("/run/cruise.conf.sh")).exists()


def test_stingray_frame_timestamps_missing_videos(fake_docker, cruise, tmp_path):
    state = stingray_frame_timestamps(cruise, FrameTimestampParams(video_data_root=str(tmp_path / "nope")),
                                      return_state=True)
    assert state.is_failed()
    assert fake_docker.calls == []


def test_stingray_image_abundance(fake_docker, cruise, tmp_path):
    classes = tmp_path / "classes.yaml"
    classes.write_text("0: copepod\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    seen = {}

    @fake_docker.on_run
    def read_config(call):
        seen["config"] = call.host_path("/run/cruise.conf.sh").read_text()

    params = ImageAbundanceParams(workspace_dir=str(tmp_path / "work"), class_yaml=str(classes),
                                  label_dirs=[str(labels)], merge_labels=True)
    state = stingray_image_abundance(cruise, params, return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.args == ["bash", "image_abundance.sh", "/run/cruise.conf.sh"]
    assert call.mount("/inputs/classes.yaml") == str(classes)
    assert call.mount("/inputs/labels/0") == str(labels)
    assert call.mount("/app/workspace") == str(tmp_path / "work")
    assert "LABEL_DIRS=(/inputs/labels/0)" in seen["config"]
    assert "MERGE_LABELS=1" in seen["config"]
