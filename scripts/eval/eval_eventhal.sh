#!/bin/bash

MODEL_PATH="$1"
shift 

python -m eval.eval_eventhal --model-path "$MODEL_PATH" "$@" --annotation-path entire_questions.json

python -m eval.eval_eventhal --model-path "$MODEL_PATH" "$@" --annotation-path interleave_questions.json

python -m eval.eval_eventhal --model-path "$MODEL_PATH" "$@" --annotation-path misleading_questions.json

