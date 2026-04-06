#!/bin/bash

PICKLE_DIR="pickles"

echo "Starting debug runs for all supported datasets..."

# bigbench_movie
# python3 main_debug.py \
#     --dataset bigbench_movie \
#     --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
#     --model llama \
#     --shot_mode zero \
#     --confidence \
#     --debug_conf \
#     --tag debug-0402-bigbench-movie \
#     --nb_dropout_samples 10 \
#     --experimental_jackknife \
#     --discord

# logiqa
python3 main_debug.py \
    --dataset logiqa \
    --pickle_path $PICKLE_DIR/logiqa.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-logiqa \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord

# college_math
python3 main_debug.py \
    --dataset college_math \
    --pickle_path $PICKLE_DIR/college_math.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-college-math \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord

# bigbench_causal
python3 main_debug.py \
    --dataset bigbench_causal \
    --pickle_path $PICKLE_DIR/bigbench_causal.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-bigbench-causal \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord

# hotpotqa
python3 main_debug.py \
    --dataset hotpotqa \
    --pickle_path $PICKLE_DIR/hotpotqa.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-hotpotqa \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord

# math500
python3 main_debug.py \
    --dataset math500 \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-math500 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord

# olympiadbench
python3 main_debug.py \
    --dataset olympiadbench \
    --pickle_path $PICKLE_DIR/olympiadbench.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-olympiadbench \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
