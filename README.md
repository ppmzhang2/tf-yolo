# YOLOv3-TensorFlow

## Demo

Data Processing:

```sh
# drop all tables
pdm run yolov3 sqlite-drop-all
# create all tables
pdm run yolov3 sqlite-create-all
# create annotation CSV files
pdm run yolov3 coco-annot-to-csv \
    --file-in-json="data/coco_instances_val2014.json" \
    --imgtag-csv="data/coco_imgtag_val2014.csv" \
    --cate-csv="data/coco_cate_val2014.csv" \
    --box-csv="data/coco_box_val2014.csv"
# load into sqlite annotation CSV files
pdm run yolov3 load-coco-annot-csv \
    --imgtag-csv="./data/coco_imgtag_val2014.csv" \
    --cate-csv="./data/coco_cate_val2014.csv" \
    --box-csv="./data/coco_box_val2014.csv"
# add base64 encoded image
pdm run yolov3 update-img-data --img-folder=data/coco_val2014
# create a single annotation / label table for YOLO training
pdm run yolov3 create-yolo-labels
```
