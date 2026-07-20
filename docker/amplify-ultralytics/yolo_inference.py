#!/usr/bin/env python3
import argparse
import sys
from math import ceil
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def chunk(lst, n):
    """Split list into n almost equal chunks preserving order."""
    if n <= 0:
        return [lst]
    length = len(lst)
    if length == 0:
        return [[] for _ in range(n)]
    size = ceil(length / n)
    return [lst[i * size : (i + 1) * size] for i in range(n)]


def parse_int_list(s):
    if s is None or s == "" or s.lower() == "none":
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_bool(s: str) -> bool:
    if s == "True":
        return True
    if s == "False":
        return False
    raise ValueError(f"Expected 'True' or 'False', got '{s}'")


def load_completed_files(manifest_file):
    """Load the set of completed file paths from the manifest."""
    if not manifest_file.exists():
        return set()
    try:
        with open(manifest_file, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        eprint(f"WARNING: Could not read manifest {manifest_file}: {e}")
        return set()


def mark_file_complete(manifest_file, file_path, lock):
    """Append a file path to the completion manifest."""
    try:
        with lock:
            with open(manifest_file, "a") as f:
                f.write(str(file_path) + "\n")
    except Exception as e:
        eprint(f"ERROR: Could not write to manifest {manifest_file}: {e}")


def relative_to_root(file_path, src_root):
    try:
        return file_path.relative_to(src_root)
    except ValueError:
        return Path(file_path.name)


def prepare_yolo_source(file_path, src_root, converted_root):
    """
    Ultralytics may load some TIFFs as one-channel tensors. Stage non-3-channel
    image files as temporary 3-channel images before passing them to YOLO.
    """
    if file_path.suffix.lower() not in IMAGE_EXTS:
        return str(file_path)

    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return str(file_path)

    if img.ndim == 3 and img.shape[2] == 3:
        return str(file_path)

    converted = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if converted is None:
        return str(file_path)

    converted_path = converted_root / relative_to_root(file_path, src_root)
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(converted_path), converted):
        raise RuntimeError(
            f"Could not write converted 3-channel image to {converted_path}"
        )

    return str(converted_path)


def validate_media_file(file_path):
    """
    Validate that an image or video file is readable by OpenCV/YOLO.

    Returns:
        tuple: (is_valid, reason, frame_count)
        frame_count is None for images, or the number of frames for videos.
    """
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "empty file (0 bytes)", None
    except Exception as e:
        return False, f"unable to stat file: {e}", None

    file_ext = file_path.suffix.lower()

    try:
        if file_ext in IMAGE_EXTS:
            img = cv2.imread(str(file_path))
            if img is None:
                return False, "cannot read image with OpenCV", None
            return True, None, None

        cap = None
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return False, "cannot open with OpenCV", None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return False, "no frames detected", None

            ret, frame = cap.read()
            if not ret or frame is None:
                return False, "cannot read first frame", None

            return True, None, frame_count
        finally:
            if cap is not None:
                cap.release()

    except Exception as e:
        return False, f"validation error: {e}", None


def process_files_on_gpu(
    gpu_id,
    files,
    completed_files,
    manifest_file,
    manifest_lock,
    args,
    classes_list,
    embed_list,
):
    from ultralytics import YOLO

    try:
        eprint(f"GPU {gpu_id}: Loading model {args.model}...")
        model = YOLO(args.model)
        eprint(f"GPU {gpu_id}: Model loaded, processing {len(files)} files")

        processed = 0
        skipped = 0
        errors = 0

        with TemporaryDirectory(prefix=f"yolo-gpu{gpu_id}-") as converted_dir:
            converted_root = Path(converted_dir)
            src_root = Path(args.source_root)

            for file_path in files:
                file_str = str(file_path)

                if file_str in completed_files:
                    skipped += 1
                    eprint(f"GPU {gpu_id}: Skipping {file_path.name} (already complete)")
                    continue

                try:
                    eprint(f"GPU {gpu_id}: Processing {file_path.name}...")
                    yolo_source = prepare_yolo_source(
                        file_path, src_root, converted_root
                    )
                    if yolo_source != file_str:
                        eprint(
                            f"GPU {gpu_id}: Converted {file_path.name} "
                            "to a temporary 3-channel image"
                        )

                    predict_kwargs = {
                        "source": yolo_source,
                        "device": gpu_id,
                        "project": args.project,
                        "name": f"gpu{gpu_id}",
                        "exist_ok": True,
                        "agnostic_nms": args.agnostic_nms,
                        "iou": args.iou,
                        "conf": args.conf,
                        "imgsz": args.imgsz,
                        "batch": args.batch,
                        "half": args.half,
                        "max_det": args.max_det,
                        "vid_stride": args.vid_stride,
                        "stream_buffer": args.stream_buffer,
                        "visualize": args.visualize,
                        "augment": args.augment,
                        "retina_masks": args.retina_masks,
                        "verbose": args.verbose,
                        "show": args.show,
                        "save": args.save,
                        "save_txt": args.save_txt,
                        "save_conf": args.save_conf,
                        "save_crop": args.save_crop,
                        "save_frames": args.save_frames,
                        "show_labels": args.show_labels,
                        "show_conf": args.show_conf,
                        "show_boxes": args.show_boxes,
                    }

                    if classes_list is not None:
                        predict_kwargs["classes"] = classes_list
                    if embed_list is not None:
                        predict_kwargs["embed"] = embed_list

                    model.predict(**predict_kwargs)

                    mark_file_complete(manifest_file, file_path, manifest_lock)
                    processed += 1
                    eprint(f"GPU {gpu_id}: Completed {file_path.name}")

                except Exception as e:
                    errors += 1
                    eprint(f"GPU {gpu_id}: ERROR processing {file_path.name}: {e}")

        eprint(
            f"GPU {gpu_id}: Finished - {processed} processed, "
            f"{skipped} skipped, {errors} errors"
        )
        if errors:
            raise RuntimeError(f"GPU {gpu_id} had {errors} per-file errors")
        if processed == 0 and skipped < len(files):
            raise RuntimeError(
                f"GPU {gpu_id} processed 0 of {len(files)} assigned files"
            )

    except Exception as e:
        eprint(f"GPU {gpu_id}: FATAL ERROR: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset across GPUs and run YOLO predict in parallel."
    )

    parser.add_argument(
        "device", type=str, help='GPU ids as comma-separated string, e.g. "0,1,2"'
    )
    parser.add_argument("agnostic_nms", type=parse_bool)
    parser.add_argument("iou", type=float)
    parser.add_argument("conf", type=float)
    parser.add_argument("imgsz", type=int)
    parser.add_argument("batch", type=int)
    parser.add_argument("half", type=parse_bool)
    parser.add_argument("max_det", type=int)
    parser.add_argument("vid_stride", type=int)
    parser.add_argument("stream_buffer", type=parse_bool)
    parser.add_argument("visualize", type=parse_bool)
    parser.add_argument("augment", type=parse_bool)
    parser.add_argument("classes", type=str, help="Comma-separated ints or 'None'")
    parser.add_argument("retina_masks", type=parse_bool)
    parser.add_argument("embed", type=str, help="Comma-separated ints or 'None'")
    parser.add_argument("name", type=str)
    parser.add_argument("verbose", type=parse_bool)
    parser.add_argument("show", type=parse_bool)
    parser.add_argument("save", type=parse_bool)
    parser.add_argument("save_frames", type=parse_bool)
    parser.add_argument("save_txt", type=parse_bool)
    parser.add_argument("save_conf", type=parse_bool)
    parser.add_argument("save_crop", type=parse_bool)
    parser.add_argument("show_labels", type=parse_bool)
    parser.add_argument("show_conf", type=parse_bool)
    parser.add_argument("show_boxes", type=parse_bool)

    parser.add_argument("--source-root", default="/data", help="Root to scan")
    parser.add_argument("--model", default="/input/weights.pt")
    parser.add_argument("--project", default="/output")
    parser.add_argument("--ext", default=".avi", help="File extension to scan")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of discovered files to process",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip OpenCV validation before inference",
    )

    return parser.parse_args()


def discover_files(args):
    src_root = Path(args.source_root)
    discovered_files = sorted([p for p in src_root.rglob(f"*{args.ext}") if p.is_file()])
    if args.max_files is not None:
        discovered_files = discovered_files[: args.max_files]
    return discovered_files


def validate_files(discovered_files, skip_validation):
    if skip_validation:
        eprint(f"Found {len(discovered_files)} files, skipping validation")
        return [(file_path, None) for file_path in discovered_files], 0

    eprint(f"Found {len(discovered_files)} files, validating...")
    valid_files_with_metadata = []
    validation_skipped = 0
    for idx, file_path in enumerate(discovered_files, start=1):
        if idx % 10000 == 0:
            eprint(f"Validated {idx}/{len(discovered_files)} files...")

        is_valid, reason, frame_count = validate_media_file(file_path)
        if is_valid:
            valid_files_with_metadata.append((file_path, frame_count))
        else:
            validation_skipped += 1
            eprint(f"WARNING: Skipping {file_path}: {reason}")

    return valid_files_with_metadata, validation_skipped


def main():
    args = parse_args()

    gpu_ids = [d.strip() for d in args.device.split(",") if d.strip()]
    if not gpu_ids:
        eprint("No GPUs provided in --device (e.g. '0,1,2').")
        return 2

    discovered_files = discover_files(args)
    if not discovered_files:
        eprint(f"No files found under {Path(args.source_root)} with extension {args.ext}")
        return 3

    valid_files_with_metadata, validation_skipped = validate_files(
        discovered_files, args.skip_validation
    )
    if not valid_files_with_metadata:
        eprint(
            f"No valid media files found. All {len(discovered_files)} files were skipped."
        )
        return 3

    if validation_skipped > 0:
        eprint(
            f"Validated: {len(valid_files_with_metadata)} valid files, "
            f"{validation_skipped} skipped"
        )
    else:
        eprint(f"Validated: All {len(valid_files_with_metadata)} files are valid")

    project_path = Path(args.project)
    project_path.mkdir(parents=True, exist_ok=True)
    manifest_file = project_path / ".completed_files.txt"

    eprint(f"Loading completion manifest from {manifest_file}...")
    completed_files = load_completed_files(manifest_file)

    files = [file_path for file_path, _ in valid_files_with_metadata]
    already_complete = sum(1 for f in files if str(f) in completed_files)

    if already_complete == len(files):
        eprint(f"All {len(files)} files already complete. Nothing to process.")
        return 0

    if already_complete > 0:
        eprint(
            f"Found {already_complete} already complete, "
            f"will process {len(files) - already_complete} files"
        )
    else:
        eprint(f"Processing all {len(files)} files")

    num_workers = min(len(gpu_ids), len(files))
    slices = chunk(files, num_workers)
    classes_list = parse_int_list(args.classes)
    embed_list = parse_int_list(args.embed)

    ctx = get_context("spawn")
    manifest_lock = ctx.Lock()

    eprint(f"Spawning {num_workers} GPU workers...")
    workers = []
    for idx in range(num_workers):
        gpu_id = gpu_ids[idx]
        file_subset = slices[idx]
        if not file_subset:
            continue

        process = ctx.Process(
            target=process_files_on_gpu,
            args=(
                gpu_id,
                file_subset,
                completed_files,
                manifest_file,
                manifest_lock,
                args,
                classes_list,
                embed_list,
            ),
        )
        process.start()
        workers.append(process)

    eprint(f"Waiting for {len(workers)} workers to complete...")
    exit_code = 0
    for process in workers:
        process.join()
        if process.exitcode != 0 and exit_code == 0:
            exit_code = process.exitcode

    if exit_code == 0:
        eprint("All workers completed successfully")
    else:
        eprint(f"One or more workers failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
