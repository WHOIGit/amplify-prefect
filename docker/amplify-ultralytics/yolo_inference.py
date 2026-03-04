#!/usr/bin/env python3
import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path
from math import ceil
import cv2

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

def write_listfile(paths, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(str(p) + "\n")

def check_outputs_complete(file_path, frame_count, output_dirs, settings):
    """
    Check if all expected outputs for a file already exist in any output directory.

    Args:
        file_path: Path to input file
        frame_count: Number of frames (for videos) or None (for images)
        output_dirs: List of output directories to check (e.g., [Path("/output/gpu0"), Path("/output/gpu1")])
        settings: Namespace with save settings (save, save_txt, save_frames, vid_stride)

    Returns:
        bool: True if all expected outputs exist and are complete, False otherwise

    Note: save_crop is not checked as it uses a flat directory structure
          making per-file verification difficult.
    """
    stem = file_path.stem
    is_video = frame_count is not None

    # If no save options are enabled, we can't check anything
    if not (settings.save or settings.save_txt or settings.save_frames):
        return False

    # Check each output directory
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        all_complete = True

        # Check save_txt outputs
        if settings.save_txt:
            if is_video:
                # For videos: count label files matching {stem}_*.txt
                expected_frames = ceil(frame_count / settings.vid_stride)
                labels_dir = output_dir / "labels"
                if labels_dir.exists():
                    label_files = list(labels_dir.glob(f"{stem}_*.txt"))
                    if len(label_files) != expected_frames:
                        all_complete = False
                else:
                    all_complete = False
            else:
                # For images: check for single label file
                label_file = output_dir / "labels" / f"{stem}.txt"
                if not label_file.exists():
                    all_complete = False

        # Check save_frames outputs (video only)
        if settings.save_frames and is_video:
            expected_frames = ceil(frame_count / settings.vid_stride)
            frames_dir = output_dir / f"{stem}_frames"
            if frames_dir.exists():
                frame_files = list(frames_dir.glob(f"{stem}_*.jpg"))
                if len(frame_files) != expected_frames:
                    all_complete = False
            else:
                all_complete = False

        # Check save outputs (annotated image/video)
        if settings.save:
            # YOLO outputs .avi for videos on Linux, original extension for images
            if is_video:
                output_file = output_dir / f"{stem}.avi"
            else:
                output_file = output_dir / f"{stem}{file_path.suffix}"

            if not output_file.exists() or output_file.stat().st_size == 0:
                all_complete = False

        # If this directory has all complete outputs, return True
        if all_complete:
            return True

    return False

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

# Check for existing outputs and filter out already-complete files
print(f"Checking for existing outputs in {args.project}...", file=sys.stderr)
project_path = Path(args.project)
output_dirs = sorted(project_path.glob("gpu*")) if project_path.exists() else []

files = []  # Final list of files to process
already_complete = 0
for file_path, frame_count in valid_files_with_metadata:
    if check_outputs_complete(file_path, frame_count, output_dirs, args):
        already_complete += 1
        print(f"Skipping {file_path}: outputs already complete", file=sys.stderr)
    else:
        files.append(file_path)

if not files:
    print(f"All {len(valid_files_with_metadata)} files have complete outputs. Nothing to process.", file=sys.stderr)
    sys.exit(0)

if already_complete > 0:
    print(f"Processing {len(files)} files, {already_complete} already complete", file=sys.stderr)
else:
    print(f"Processing all {len(files)} files", file=sys.stderr)

num_workers = min(len(gpu_ids), len(files))
slices = chunk(files, num_workers)

classes_list = parse_int_list(args.classes)
embed_list = parse_int_list(args.embed)

procs = []
for idx in range(num_workers):
    dev = gpu_ids[idx]
    subset = slices[idx]
    if not subset:
        continue

    listfile = Path(args.project) / f"image_list.gpu{dev}.txt"
    write_listfile(subset, listfile)
    run_name = f"gpu{dev}"

    cmd = [
        "yolo",
        "mode=predict",
        "task=detect",
        f"source={listfile}",
        f"model={args.model}",
        f"project={args.project}",
        f"device={dev}",
        f"agnostic_nms={args.agnostic_nms}",
        f"iou={args.iou}",
        f"conf={args.conf}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"half={args.half}",
        f"max_det={args.max_det}",
        f"vid_stride={args.vid_stride}",
        f"stream_buffer={args.stream_buffer}",
        f"visualize={args.visualize}",
        f"augment={args.augment}",
        f"retina_masks={args.retina_masks}",
        f"name={run_name}",
        f"verbose={args.verbose}",
        f"show={args.show}",
        f"save={args.save}",
        f"save_txt={args.save_txt}",
        f"save_conf={args.save_conf}",
        f"save_crop={args.save_crop}",
        f"show_labels={args.show_labels}",
        f"show_conf={args.show_conf}",
        f"show_boxes={args.show_boxes}",
    ]

    if classes_list is not None:
        cmd.append(f"classes={classes_list}")
    if embed_list is not None:
        cmd.append(f"embed={embed_list}")

    print("Launching:", " ".join(shlex.quote(c) for c in cmd))
    procs.append(subprocess.Popen(cmd))

exit_code = 0
for p in procs:
    rc = p.wait()
    if rc != 0 and exit_code == 0:
        exit_code = rc

sys.exit(exit_code)

