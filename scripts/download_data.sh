#!/bin/bash
# Datasets for the detector, run on the cluster login node after scripts/setup_cluster.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(python3 slurm/generate.py --shell-env)"
DATA=$(eval echo "$DATA")
mkdir -p "$DATA/coco"
# COCO 2017 images (about 19 GB); the scripts fetch the annotations themselves
for split in train2017 val2017; do
    if [[ ! -d $DATA/coco/$split ]]; then
        curl -L "http://images.cocodataset.org/zips/$split.zip" -o "$DATA/coco/$split.zip"
        (cd "$DATA/coco" && unzip -q "$split.zip" && rm "$split.zip")
    fi
done
python3 ml/data/openimages_pool.py --out "$DATA/openimages_pool"
python3 ml/data/coco_pool.py --coco "$DATA/coco" --out "$DATA/coco_pool"
python3 ml/data/coco_objects.py --coco "$DATA/coco" --out "$DATA/coco_objects"
