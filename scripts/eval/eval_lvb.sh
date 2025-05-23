#!/bin/bash

MODEL_PATH="$1"
shift 

$SCRATCH/pytorch-example/python -m eval.eval_lvb --model-path "$MODEL_PATH" "$@" --duration 15 
$SCRATCH/pytorch-example/python -m eval.eval_lvb --model-path "$MODEL_PATH" "$@" --duration 60 
$SCRATCH/pytorch-example/python -m eval.eval_lvb --model-path "$MODEL_PATH" "$@" --duration 600
$SCRATCH/pytorch-example/python -m eval.eval_lvb --model-path "$MODEL_PATH" "$@" --duration 3600

