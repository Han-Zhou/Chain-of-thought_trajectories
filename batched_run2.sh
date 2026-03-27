#!/bin/bash

echo "Starting batched run 2..."



# few-shot, type 2
python3 main.py \
    --dataset logiqa \
    --model qwen \
    --sample_size 10 \
    --shot_mode few \
    --thinking \
    --max_new_tokens 20000 \
    --type 2 \
    --debug_conf \
    --confidence \
    --sample_range 300 652

# # few-shot, no-thinking
# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode few \
#     --max_new_tokens 20000 \
#     --type 1 \
#     --debug_conf \
#     --confidence



# # zero-shot, type 2
# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode zero \
#     --thinking \
#     --max_new_tokens 20000 \
#     --type 2 \
#     --debug_conf \
#     --confidence


# # zero-shot, no-thinking
# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode zero \
#     --max_new_tokens 20000 \
#     --type 1 \
#     --debug_conf \
#     --confidence




# # few-shot, type 1
# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode few \
#     --thinking \
#     --max_new_tokens 20000 \
#     --type 1 \
#     --debug_conf \
#     --confidence

# # zero-shot, type 1
# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode zero \
#     --thinking \
#     --max_new_tokens 20000 \
#     --type 1 \
#     --debug_conf \
#     --confidence
