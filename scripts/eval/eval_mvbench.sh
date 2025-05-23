#!/bin/bash

MODEL_PATH="$1"
shift 

python -m eval.eval_mvbench --model-path "$MODEL_PATH" "$@"

 

