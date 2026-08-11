#!/usr/bin/env python3
import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from multiprocessing import get_context
from os import cpu_count
from pathlib import Path
from tempfile import gettempdir, mkdtemp

import cv2

STAGING_PREFIX = "yolo-gpu"

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


def batched(items, size):
    """Yield successive lists of at most `size` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


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


def mark_files_complete(manifest_file, file_paths, lock):
    """Append a batch of completed file paths to the completion manifest."""
    if not file_paths:
        return
    payload = "".join(f"{file_path}\n" for file_path in file_paths)
    try:
        with lock:
            with open(manifest_file, "a") as f:
                f.write(payload)
                f.flush()
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


def resolve_validation_workers(validation_workers, file_count):
    if file_count <= 0:
        return 1
    if validation_workers and validation_workers > 0:
        return min(validation_workers, file_count)
    return min(file_count, max(1, min(cpu_count() or 1, 32)))


def build_predict_kwargs(gpu_id, args, classes_list, embed_list):
    """Build the keyword arguments shared by every predict call on this GPU."""
    predict_kwargs = {
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

    return predict_kwargs


def stage_batch(files, src_root, converted_root, workers, gpu_id):
    """
    Stage a batch of files for YOLO, converting non-3-channel images as needed.

    Returns:
        tuple: (sources, source_to_file, staging_errors) where sources is the list
        of paths to hand to YOLO and source_to_file maps each back to its original.
    """

    def stage_one(file_path):
        try:
            return file_path, prepare_yolo_source(file_path, src_root, converted_root), None
        except Exception as e:
            return file_path, None, e

    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as executor:
            staged = list(executor.map(stage_one, files))
    else:
        staged = [stage_one(file_path) for file_path in files]

    sources = []
    source_to_file = {}
    staging_errors = []
    converted = 0

    for file_path, source, error in staged:
        if error is not None:
            eprint(f"GPU {gpu_id}: ERROR staging {file_path.name}: {error}")
            staging_errors.append(file_path)
            continue
        sources.append(source)
        source_to_file[source] = file_path
        if source != str(file_path):
            converted += 1

    if converted:
        eprint(
            f"GPU {gpu_id}: Converted {converted}/{len(files)} images "
            "to temporary 3-channel copies"
        )

    return sources, source_to_file, staging_errors


def run_predict(model, source, predict_kwargs):
    """
    Run one predict call, consuming the streaming generator.

    Streaming keeps Results (which hold a full-size orig_img each) from
    accumulating in memory across a large batch.
    """
    seen = 0
    for _ in model.predict(source=source, stream=True, **predict_kwargs):
        seen += 1
    return seen


class StagedBatch:
    """
    A batch staged on disk and ready to hand to predict().

    Staging happens on a prefetch thread while the GPU works on the previous
    batch, so a staged batch outlives the call that created it. That rules out
    a `with TemporaryDirectory(...)` scope: whoever consumes the batch owns
    cleanup() and must call it. Each staged batch is roughly 12 MB per file on
    disk, so failing to release one promptly is expensive.
    """

    def __init__(self, files, tmpdir, listing, sources, source_to_file, staging_errors):
        self.files = files
        self.tmpdir = tmpdir
        self.listing = listing
        self.sources = sources
        self.source_to_file = source_to_file
        self.staging_errors = staging_errors

    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def sweep_stale_staging_dirs(gpu_id):
    """Remove staging directories orphaned by a previously killed container."""
    removed = 0
    for stale in Path(gettempdir()).glob(f"{STAGING_PREFIX}{gpu_id}-*"):
        shutil.rmtree(stale, ignore_errors=True)
        removed += 1
    if removed:
        eprint(f"GPU {gpu_id}: Removed {removed} stale staging directories")


def stage_batch_to_disk(files, src_root, gpu_id, convert_workers):
    """
    Stage one batch into a fresh temporary directory.

    Runs on the prefetch thread. On any failure the partial directory is
    removed before propagating, so a failed batch cannot leak ~12 GB.
    """
    tmpdir = Path(mkdtemp(prefix=f"{STAGING_PREFIX}{gpu_id}-"))
    try:
        converted_root = tmpdir / "converted"
        converted_root.mkdir()

        sources, source_to_file, staging_errors = stage_batch(
            files, src_root, converted_root, convert_workers, gpu_id
        )

        listing = tmpdir / "sources.txt"
        listing.write_text("".join(f"{source}\n" for source in sources))

        return StagedBatch(
            files, tmpdir, listing, sources, source_to_file, staging_errors
        )
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def run_staged_batch(model, staged, predict_kwargs, gpu_id, manifest_file, manifest_lock):
    """
    Run a single predict call over an already-staged batch.

    Ultralytics recomputes its "N labels saved" summary by globbing the whole
    output directory once per predict call, so batching files into one call is
    what keeps that cost from scaling with the number of files. The batch is
    handed over as a .txt listing rather than a Python list: a list source is
    routed to LoadPilAndNumpy, which reads every image into memory and treats
    the whole list as one GPU batch, while a .txt is expanded by
    LoadImagesAndVideos with normal `batch` streaming and video support.

    Returns:
        tuple: (processed, errors)
    """
    errors = len(staged.staging_errors)

    if not staged.sources:
        return 0, errors

    try:
        run_predict(model, str(staged.listing), predict_kwargs)
        completed = [staged.source_to_file[source] for source in staged.sources]
    except Exception as e:
        eprint(
            f"GPU {gpu_id}: Batch of {len(staged.sources)} files failed ({e}); "
            "retrying file by file"
        )
        completed = []
        for source in staged.sources:
            file_path = staged.source_to_file[source]
            try:
                run_predict(model, source, predict_kwargs)
                completed.append(file_path)
            except Exception as file_error:
                errors += 1
                eprint(f"GPU {gpu_id}: ERROR processing {file_path.name}: {file_error}")

    mark_files_complete(manifest_file, completed, manifest_lock)
    return len(completed), errors


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
        sweep_stale_staging_dirs(gpu_id)

        eprint(f"GPU {gpu_id}: Loading model {args.model}...")
        model = YOLO(args.model)

        pending = [f for f in files if str(f) not in completed_files]
        skipped = len(files) - len(pending)
        eprint(
            f"GPU {gpu_id}: Model loaded, processing {len(pending)} files "
            f"({skipped} already complete)"
        )

        predict_kwargs = build_predict_kwargs(gpu_id, args, classes_list, embed_list)
        src_root = Path(args.source_root)
        files_per_call = max(1, args.files_per_call)
        batches = list(batched(pending, files_per_call))
        total_batches = len(batches)

        processed = 0
        errors = 0

        # Stage batch N+1 on a background thread while the GPU runs batch N.
        # cv2 and CUDA both release the GIL, so the two genuinely overlap.
        # Depth is fixed at one: staging is faster than inference once it runs
        # concurrently, and each batch in flight costs ~12 MB per file on disk.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"stage-gpu{gpu_id}"
        ) as stager:

            def submit(index):
                if index >= total_batches:
                    return None
                return stager.submit(
                    stage_batch_to_disk,
                    batches[index],
                    src_root,
                    gpu_id,
                    args.convert_workers,
                )

            ahead = submit(0)

            for index, batch in enumerate(batches, start=1):
                eprint(
                    f"GPU {gpu_id}: Batch {index}/{total_batches} ({len(batch)} files)..."
                )

                try:
                    staged = ahead.result()
                except Exception as e:
                    # A staging failure must not abort a multi-hour run.
                    eprint(f"GPU {gpu_id}: ERROR staging batch {index}: {e}")
                    errors += len(batch)
                    ahead = submit(index)
                    continue

                # Kick off the next batch before touching the GPU; this call is
                # what creates the overlap.
                ahead = submit(index)

                try:
                    batch_processed, batch_errors = run_staged_batch(
                        model,
                        staged,
                        predict_kwargs,
                        gpu_id,
                        manifest_file,
                        manifest_lock,
                    )
                    processed += batch_processed
                    errors += batch_errors
                finally:
                    staged.cleanup()

                eprint(
                    f"GPU {gpu_id}: Batch {index}/{total_batches} done - "
                    f"{processed}/{len(pending)} files complete, {errors} errors"
                )

        eprint(
            f"GPU {gpu_id}: Finished - {processed} processed, "
            f"{skipped} skipped, {errors} errors"
        )
        if errors:
            raise RuntimeError(f"GPU {gpu_id} had {errors} per-file errors")
        if processed == 0 and pending:
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
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=0,
        help="Parallel OpenCV validation workers; 0 chooses an automatic worker count",
    )
    parser.add_argument(
        "--files-per-call",
        type=int,
        default=1000,
        help=(
            "Files handed to each YOLO predict call. Ultralytics rescans the whole "
            "output directory once per call, so larger values amortize that cost"
        ),
    )
    parser.add_argument(
        "--convert-workers",
        type=int,
        default=8,
        help="Threads used to stage 3-channel image copies before inference",
    )

    return parser.parse_args()


def discover_files(args):
    src_root = Path(args.source_root)
    discovered_files = sorted([p for p in src_root.rglob(f"*{args.ext}") if p.is_file()])
    if args.max_files is not None:
        discovered_files = discovered_files[: args.max_files]
    return discovered_files


def validate_one_file(file_path):
    is_valid, reason, frame_count = validate_media_file(file_path)
    return file_path, is_valid, reason, frame_count


def validate_files(discovered_files, skip_validation, validation_workers):
    if skip_validation:
        eprint(f"Found {len(discovered_files)} files, skipping validation")
        return [(file_path, None) for file_path in discovered_files], 0

    workers = resolve_validation_workers(validation_workers, len(discovered_files))
    eprint(f"Found {len(discovered_files)} files, validating with {workers} workers...")
    valid_files_with_metadata = []
    validation_skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(validate_one_file, discovered_files)

        for idx, (file_path, is_valid, reason, frame_count) in enumerate(results, start=1):
            if idx % 10000 == 0:
                eprint(f"Validated {idx}/{len(discovered_files)} files...")

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
        discovered_files, args.skip_validation, args.validation_workers
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
