#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on olympiadbench and college_math..."



CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --pickle_path $PICKLE_DIR/olympiadbench.pkl \
    --dataset olympiadbench \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 415-qwen-olympiadbench \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots &

CUDA_VISIBLE_DEVICES=2,3 python3 main.py \
    --pickle_path $PICKLE_DIR/college_math.pkl \
    --dataset college_math \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 415-qwen-college_math \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots &

wait
