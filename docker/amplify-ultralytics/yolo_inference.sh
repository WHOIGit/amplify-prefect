#!/bin/bash

# $1 : device: str
# $2 : agnostic_nms: bool
# $3 : iou: float
# $4 : conf: float
# $5 : imgsz: int
# $6 : batch: int
# $7 : half: bool
# $8 : max_det: int
# $9 : vid_stride: int
# $10 : stream_buffer: bool
# $11 : visualize: bool
# $12 : augment: bool
# $13 : classes: list[int]
# $14 : retina_masks: bool
# $15 : embed: list[int]
# $16 : name: str
# $17 : verbose: bool
# $18 : show: bool
# $19 : save: bool
# $20 : save_frames: bool
# $21 : save_txt: bool 
# $22 : save_conf: bool 
# $23 : save_crop: bool 
# $24 : show_labels: bool 
# $25 : show_conf: bool 
# $26 : show_boxes: bool

find /data -type f -iname "*.avi" > image_list.txt
yolo mode=predict \
    task=detect \
    source="image_list.txt" \
    model="/input/weights.pt" \
    project="/output" \
    device=${1} \
    agnostic_nms=${2} \
    iou=${3} \
    conf=${4} \
    imgsz=${5} \
    batch=${6} \
    half=${7} \
    max_det=${8} \
    vid_stride=${9} \
    stream_buffer=${10} \
    visualize=${11} \
    augment=${12} \
    classes=${13} \
    retina_masks=${14} \
    embed=${15} \
    name=${16} \
    verbose=${17} \
    show=${18} \
    save=${19} \
    save_txt=${21} \
    save_conf=${22} \
    save_crop=${23} \
    show_labels=${24} \
    show_conf=${25} \
    show_boxes=${26}
