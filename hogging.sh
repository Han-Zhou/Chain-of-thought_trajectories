#!/bin/bash

salloc \
    --job-name=verbal-skillflow-devstral \
    --gres=gpu:2 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=7:00:00 \
    --qos=high \
    --partition=compute
    