#!/usr/bin/env python3
import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path
from math import ceil

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
files = sorted([p for p in src_root.rglob(f"*{args.ext}") if p.is_file()])
if not files:
    print(f"No files found under {src_root} with extension {args.ext}", file=sys.stderr)
    sys.exit(3)

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

