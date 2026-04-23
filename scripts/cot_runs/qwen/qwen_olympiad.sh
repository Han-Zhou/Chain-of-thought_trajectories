# #!/bin/bash

# PICKLE_DIR="pickles"

# echo "Running cot_run qwen on olympiadbench..."

# python3 main.py \
#     --pickle_path $PICKLE_DIR/olympiadbench.pkl \
#     --dataset olympiadbench \
#     --model qwen \
#     --thinking \
#     --shot_mode zero \
#     --max_new_tokens 15000 \
#     --type 1 \
#     --confidence \
#     --vanilla_only \
#     --temperature 0.9 \
#     --debug_conf \
#     --tag cots-420-qwen-olympiad-32 \
#     --experimental_extra_box \
#     --nb_cot_samples 32 \
#     --discord
