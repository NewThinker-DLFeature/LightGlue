#!/usr/bin/env bash

input=$1

dataset=/slam/runtime/heightmap_datasets/shaking_test/2024-08-06_09-01-42__gain128_exposure10
input=$dataset/color/data

# input=6

./demo_lightglue.py \
    --input $input \
    --resize -1 \
    --show_keypoints

