#!/bin/bash

MODEL_PATH="$1"
shift 

$SCRATCH/pytorch-example/python -m eval.eval_mme --model-path "$MODEL_PATH" "$@" 


