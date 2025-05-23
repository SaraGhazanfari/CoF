#!/bin/bash
MODEL_PATH="$1"
shift 


python -m eval.eval_nextqa \
    --model-path "$MODEL_PATH" "$@" --split val --part 1
python -m eval.eval_nextqa \
    --model-path "$MODEL_PATH" "$@" --split val --part 2
python -m eval.eval_nextqa \
    --model-path "$MODEL_PATH" "$@" --split val --part 3


# $SCRATCH/pytorch-example/python -m eval.eval_nextqa \
#     --model-path "$MODEL_PATH" "$@" --split test --part 1
# $SCRATCH/pytorch-example/python -m eval.eval_nextqa \
#     --model-path "$MODEL_PATH" "$@" --split test --part 2
# $SCRATCH/pytorch-example/python -m eval.eval_nextqa \
#     --model-path "$MODEL_PATH" "$@" --split test --part 3

