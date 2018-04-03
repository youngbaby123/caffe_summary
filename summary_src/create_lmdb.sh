#!/usr/bin/env sh
OUT=/home/zkyang/Workspace/DL_code/Caffe_code/caffe/data/chebiao
DATA_ROOT=/home/zkyang/Workspace/DL_code/Caffe_code/caffe/data/chebiao
TOOLS=/home/zkyang/Workspace/DL_code/Caffe_code/caffe/build/tools

# 这里我们打开resize，需要把所有图片尺寸统一
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=112
  RESIZE_WIDTH=112
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

.......

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT/Data/ \
    $DATA_ROOT/train.txt \
    $OUT/chebiao_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT/Data/ \
    $DATA_ROOT/val.txt \
    $OUT/chebiao_val_lmdb
echo "Done."
