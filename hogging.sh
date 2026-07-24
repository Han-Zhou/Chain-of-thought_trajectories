#!/bin/bash

salloc \
    --job-name=cs1qa \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=8:00:00 \
    --qos=high \
    --partition=compute
    