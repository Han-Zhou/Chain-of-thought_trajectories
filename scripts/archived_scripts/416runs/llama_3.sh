#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on bfcl, olympiadbench and college_math..."

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --pickle_path $PICKLE_DIR/bfcl.pkl \
    --dataset bfcl \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 415-llama-bfcl \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots &

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --pickle_path $PICKLE_DIR/olympiadbench.pkl \
    --dataset olympiadbench \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 415-llama-olympiadbench \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots &

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --pickle_path $PICKLE_DIR/college_math.pkl \
    --dataset college_math \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 415-llama-college_math \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots &

wait
