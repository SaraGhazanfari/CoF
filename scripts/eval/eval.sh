######## MME
bash scripts/eval/eval_mme.sh $model_path
bash scripts/eval/eval_mme.sh $model_path --cot

######## EventHal
bash scripts/eval/eval_eventhal.sh $model_path 
bash scripts/eval/eval_eventhal.sh $model_path --cot
 
######## VidHal
bash scripts/eval/eval_vidhal.sh $model_path
bash scripts/eval/eval_vidhal.sh $model_path --cot

######## MVBench
bash scripts/eval/eval_mvbench.sh $model_path 
bash scripts/eval/eval_mvbench.sh $model_path --cot

######## VSI
bash scripts/eval/eval_vsi.sh $model_path 
bash scripts/eval/eval_vsi.sh $model_path --cot

######## NextQA 
bash scripts/eval/eval_nextqa.sh $model_path
bash scripts/eval/eval_nextqa.sh $model_path --cot






