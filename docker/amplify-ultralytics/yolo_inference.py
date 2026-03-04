#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from math import ceil
import cv2
from multiprocessing import Process
from ultralytics import YOLO
import threading

# ---- helpers ----
def chunk(lst, n):
    """Split list into n (almost) equal chunks preserving order."""
    if n <= 0:
        return [lst]
    L = len(lst)
    if L == 0:
        return [[] for _ in range(n)]
    size = ceil(L / n)
    return [lst[i*size:(i+1)*size] for i in range(n)]

def parse_int_list(s):
    if s is None or s == "" or s.lower() == "none":
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]

def parse_bool(s: str) -> bool:
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError(f"Expected 'True' or 'False', got '{s}'")

def load_completed_files(manifest_file):
    """Load the set of completed file paths from the manifest."""
    if not manifest_file.exists():
        return set()
    try:
        with open(manifest_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"WARNING: Could not read manifest {manifest_file}: {e}", file=sys.stderr)
        return set()

def mark_file_complete(manifest_file, file_path, lock):
    """Append a file path to the completion manifest (thread-safe)."""
    try:
        with lock:
            with open(manifest_file, 'a') as f:
                f.write(str(file_path) + "\n")
    except Exception as e:
        print(f"ERROR: Could not write to manifest {manifest_file}: {e}", file=sys.stderr)

def validate_media_file(file_path):
    """
    Validate that an image or video file is readable by OpenCV/YOLO.
    Handles both static images (jpg, png, etc.) and video files (avi, mp4, etc.).

    Returns:
        tuple: (is_valid: bool, reason: str or None, frame_count: int or None)
        frame_count is None for images, or the number of frames for videos
    """
    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "empty file (0 bytes)", None
    except Exception as e:
        return False, f"unable to stat file: {e}", None

    # Common image extensions - use cv2.imread for these
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_ext = file_path.suffix.lower()

    try:
        if file_ext in image_exts:
            # For images, try to read with cv2.imread
            img = cv2.imread(str(file_path))
            if img is None:
                return False, "cannot read image with OpenCV", None
            return True, None, None
        else:
            # For videos or other formats, use VideoCapture
            cap = None
            try:
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    return False, "cannot open with OpenCV", None

                # Check frame count
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    return False, "no frames detected", None

                # Try to read first frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    return False, "cannot read first frame", None

                return True, None, frame_count

            finally:
                if cap is not None:
                    cap.release()

    except Exception as e:
        return False, f"validation error: {e}", None

def process_files_on_gpu(gpu_id, files, completed_files, manifest_file, manifest_lock, args, classes_list, embed_list):
    """
    Worker function: Load YOLO model once and process all assigned files on one GPU.

    Args:
        gpu_id: GPU device ID (e.g., "0", "1", "2")
        files: List of file paths to process
        completed_files: Set of already-completed file paths
        manifest_file: Path to completion manifest file
        manifest_lock: Threading lock for manifest writes
        args: Argument namespace with all YOLO parameters
        classes_list: Parsed list of class IDs or None
        embed_list: Parsed list of embed layers or None
    """
    try:
        # Load model once for this GPU
        print(f"GPU {gpu_id}: Loading model {args.model}...", file=sys.stderr)
        model = YOLO(args.model)
        print(f"GPU {gpu_id}: Model loaded, processing {len(files)} files", file=sys.stderr)

        processed = 0
        skipped = 0
        errors = 0

        for file_path in files:
            file_str = str(file_path)

            # Skip if already complete
            if file_str in completed_files:
                skipped += 1
                print(f"GPU {gpu_id}: Skipping {file_path.name} (already complete)", file=sys.stderr)
                continue

            try:
                print(f"GPU {gpu_id}: Processing {file_path.name}...", file=sys.stderr)

                # Build prediction arguments
                predict_kwargs = {
                    'source': file_str,
                    'device': gpu_id,
                    'project': args.project,
                    'name': f'gpu{gpu_id}',
                    'exist_ok': True,
                    'agnostic_nms': args.agnostic_nms,
                    'iou': args.iou,
                    'conf': args.conf,
                    'imgsz': args.imgsz,
                    'batch': args.batch,
                    'half': args.half,
                    'max_det': args.max_det,
                    'vid_stride': args.vid_stride,
                    'stream_buffer': args.stream_buffer,
                    'visualize': args.visualize,
                    'augment': args.augment,
                    'retina_masks': args.retina_masks,
                    'verbose': args.verbose,
                    'show': args.show,
                    'save': args.save,
                    'save_txt': args.save_txt,
                    'save_conf': args.save_conf,
                    'save_crop': args.save_crop,
                    'save_frames': args.save_frames,
                    'show_labels': args.show_labels,
                    'show_conf': args.show_conf,
                    'show_boxes': args.show_boxes,
                }

                # Add optional parameters
                if classes_list is not None:
                    predict_kwargs['classes'] = classes_list
                if embed_list is not None:
                    predict_kwargs['embed'] = embed_list

                # Run prediction
                results = model.predict(**predict_kwargs)

                # Mark as complete
                mark_file_complete(manifest_file, file_path, manifest_lock)
                processed += 1
                print(f"GPU {gpu_id}: Completed {file_path.name}", file=sys.stderr)

            except Exception as e:
                errors += 1
                print(f"GPU {gpu_id}: ERROR processing {file_path.name}: {e}", file=sys.stderr)
                # Continue to next file instead of crashing

        print(f"GPU {gpu_id}: Finished - {processed} processed, {skipped} skipped, {errors} errors", file=sys.stderr)

    except Exception as e:
        print(f"GPU {gpu_id}: FATAL ERROR: {e}", file=sys.stderr)
        raise

# ---- args ----
parser = argparse.ArgumentParser(
    description="Split dataset across GPUs and run YOLO predict in parallel."
)

parser.add_argument("device", type=str, help='GPU ids as comma-separated string, e.g. "0,1,2"')

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

parser.add_argument("--source-root", default="/data", help="Root to scan for .avi files")
parser.add_argument("--model", default="/input/weights.pt")
parser.add_argument("--project", default="/output")
parser.add_argument("--ext", default=".avi", help="File extension to scan (default: .avi)")

args = parser.parse_args()

# ---- core ----
gpu_ids = [d.strip() for d in args.device.split(",") if d.strip()]
if not gpu_ids:
    print("No GPUs provided in --device (e.g. '0,1,2').", file=sys.stderr)
    sys.exit(2)

src_root = Path(args.source_root)
discovered_files = sorted([p for p in src_root.rglob(f"*{args.ext}") if p.is_file()])
if not discovered_files:
    print(f"No files found under {src_root} with extension {args.ext}", file=sys.stderr)
    sys.exit(3)

# Validate media files before processing
print(f"Found {len(discovered_files)} files, validating...", file=sys.stderr)
valid_files_with_metadata = []  # List of (file_path, frame_count) tuples
validation_skipped = 0
for file_path in discovered_files:
    is_valid, reason, frame_count = validate_media_file(file_path)
    if is_valid:
        valid_files_with_metadata.append((file_path, frame_count))
    else:
        validation_skipped += 1
        print(f"WARNING: Skipping {file_path}: {reason}", file=sys.stderr)

if not valid_files_with_metadata:
    print(f"No valid media files found. All {len(discovered_files)} files were skipped.", file=sys.stderr)
    sys.exit(3)

if validation_skipped > 0:
    print(f"Validated: {len(valid_files_with_metadata)} valid files, {validation_skipped} skipped", file=sys.stderr)
else:
    print(f"Validated: All {len(valid_files_with_metadata)} files are valid", file=sys.stderr)

# Load completion manifest
project_path = Path(args.project)
project_path.mkdir(parents=True, exist_ok=True)
manifest_file = project_path / ".completed_files.txt"
manifest_lock = threading.Lock()

print(f"Loading completion manifest from {manifest_file}...", file=sys.stderr)
completed_files = load_completed_files(manifest_file)

# Extract just file paths (drop frame_count metadata)
files = [file_path for file_path, _ in valid_files_with_metadata]

# Count how many are already complete
already_complete = sum(1 for f in files if str(f) in completed_files)

if already_complete == len(files):
    print(f"All {len(files)} files already complete. Nothing to process.", file=sys.stderr)
    sys.exit(0)

if already_complete > 0:
    print(f"Found {already_complete} already complete, will process {len(files) - already_complete} files", file=sys.stderr)
else:
    print(f"Processing all {len(files)} files", file=sys.stderr)

# Distribute files across GPUs
num_workers = min(len(gpu_ids), len(files))
slices = chunk(files, num_workers)

# Parse optional class and embed lists
classes_list = parse_int_list(args.classes)
embed_list = parse_int_list(args.embed)

# Spawn one worker process per GPU
print(f"Spawning {num_workers} GPU workers...", file=sys.stderr)
workers = []
for idx in range(num_workers):
    gpu_id = gpu_ids[idx]
    file_subset = slices[idx]
    if not file_subset:
        continue

    p = Process(
        target=process_files_on_gpu,
        args=(gpu_id, file_subset, completed_files, manifest_file, manifest_lock, args, classes_list, embed_list)
    )
    p.start()
    workers.append(p)

# Wait for all workers to complete
print(f"Waiting for {len(workers)} workers to complete...", file=sys.stderr)
exit_code = 0
for p in workers:
    p.join()
    if p.exitcode != 0 and exit_code == 0:
        exit_code = p.exitcode

if exit_code == 0:
    print("All workers completed successfully", file=sys.stderr)
else:
    print(f"One or more workers failed with exit code {exit_code}", file=sys.stderr)

sys.exit(exit_code)

