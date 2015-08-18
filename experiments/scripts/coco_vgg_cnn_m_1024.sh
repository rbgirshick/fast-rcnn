#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_baseline_vgg_cnn_m_1024.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG_CNN_M_1024/coco/solver.prototxt \
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
  --cfg experiments/cfgs/coco.yml --imdb coco_2014_train \
  --iters 280000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG_CNN_M_1024/coco/test.prototxt \
  --net output/coco_baseline/coco_2014_train/vgg_cnn_m_1024_fast_rcnn_iter_280000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minival \
  --num_dets 100
