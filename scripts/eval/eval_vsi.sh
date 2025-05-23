#!/bin/bash


MODEL_PATH="$1"
shift 


python -m eval.eval_vsi \
 --task object_counting \
 --model-path "$MODEL_PATH" "$@"

python -m eval.eval_vsi \
--task object_size_estimation \
--model-path "$MODEL_PATH" "$@"

python -m eval.eval_vsi \
--task room_size_estimation \
--model-path "$MODEL_PATH" "$@"

python -m eval.eval_vsi \
--task object_abs_distance \
--model-path "$MODEL_PATH" "$@"

 #-------------------------------------------------

 python -m eval.eval_vsi \
 --task obj_appearance_order \
 --model-path "$MODEL_PATH" "$@"

 python -m eval.eval_vsi \
 --task object_rel_distance \
 --model-path "$MODEL_PATH" "$@"

 python -m eval.eval_vsi \
 --task object_rel_direction \
 --model-path "$MODEL_PATH" "$@"

 python -m eval.eval_vsi \
 --task route_planning \
 --model-path "$MODEL_PATH" "$@"

