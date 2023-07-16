# YOLOv3-TensorFlow

## Environment

Install TensorFlow 2.0 and other dependencies:

```bash
# CUDA 11.8 + cuDNN 8.6 + Python 3.10
conda env create -f conda-cu11-py310.yaml
# Apple Silicon + Python 3.11
conda env create -f conda-apple-py311.yaml
```

Uninstall:

```bash
conda env remove --name tf-rcnn-cu11-py310 --all
```

## Demo

Data Processing:

```sh
# drop all tables
yolov3 sqlite-drop-all
# create all tables
yolov3 sqlite-create-all
# create annotation CSV files
yolov3 coco-annot-to-csv \
    --file-in-json="data/coco_instances_val2014.json" \
    --imgtag-csv="data/coco_imgtag_val2014.csv" \
    --cate-csv="data/coco_cate_val2014.csv" \
    --box-csv="data/coco_box_val2014.csv"
# load into sqlite annotation CSV files
yolov3 load-coco-annot-csv \
    --imgtag-csv="./data/coco_imgtag_val2014.csv" \
    --cate-csv="./data/coco_cate_val2014.csv" \
    --box-csv="./data/coco_box_val2014.csv"
# add base64 encoded image
yolov3 update-img-data --img-folder=data/coco_val2014
# create a single annotation / label table for YOLO training
yolov3 create-yolo-labels
```
