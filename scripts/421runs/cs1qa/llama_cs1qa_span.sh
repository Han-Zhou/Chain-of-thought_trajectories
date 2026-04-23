#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on cs1qa_span subset..."


python3 main.py \
    --pickle_path $PICKLE_DIR/cs1qa_span.pkl \
    --dataset cs1qa_span \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 421-llama-cs1qa-span-subset \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --sample_size 100 \
    --discord



