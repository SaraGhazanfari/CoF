import os 
import json
import argparse
import time, torch
from eval.utils import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='OpenGVLab/InternVL2_5-1B')
    parser.add_argument('--annotation-path', type=str, default='entire_questions.json')    # entire_questions.json  misleading_questions.json  
    parser.add_argument('--video-path', type=str, default='/vast/sg7457/video_datasets/hal/EventHallusion/videos/')
    parser.add_argument('--cot', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    ann_root = '/vast/sg7457/video_datasets/hal/EventHallusion/questions/'
    with open(os.path.join(ann_root, args.annotation_path)) as file:
        ann = json.load(file)
        
    if 'intern' in args.model_path.lower():
        model, tokenizer, generation_config = load_model(args.model_path, max_new_tokens=2048)
    
    elif 'llava' in args.model_path.lower(): 
        model, processor = load_llava_next_model(args.model_path)
    
    else:
        model, processor = load_qwen_model(args.model_path)
        
    start_t = time.time()
    count, all = 0, 0
    for sample_idx, sample in enumerate(ann):
        video_name = sample['id']
        if len(video_name) == 0:
            num = sample_idx + 1
            if num < 10:
                num = f'00{num}'
            elif num < 100:
                num = f'0{num}'
            video_name = f'interleave_{num}'
        video_path = os.path.join(args.video_path, args.annotation_path.split('_')[0], video_name+'.mp4')
        
        
        for question_dict in sample['questions']:
            prompt = question_dict['question']
            if args.cot:
                question = add_cot_to_question(prompt, binary=True)
            else:
                question = prompt + '\n answer with Yes or No'
            if 'intern' in args.model_path.lower():
                pixel_values, num_patches_list = load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=30)
                response = generate_response_for_video(pixel_values, num_patches_list, question, model, tokenizer, generation_config)

            elif 'llava' in args.model_path.lower():
                response = get_llava_next_response(video_path, question, model, processor)
            
            else:
                fps=1.0#0.5
                # if '72B' in args.model_path:
                #     fps=0.5
                response = get_qwen_vl_response(video_path, question, model, processor, fps=fps)

            
            interval = round((time.time() - start_t)/60, 2)
            
            if question_dict['answer'].replace('.', '').lower() == 'yes' and 'yes' in response.lower():
                count += 1
            if question_dict['answer'].replace('.', '').lower() == 'no' and 'yes' not in response.lower() and 'no' in response.lower():
                count +=1

            all += 1

            print(f"{sample_idx}/{len(ann)}") #, Time: {interval} Q: {question}")
            print(f"GT:{question_dict['answer']}")
            print('=====================')
            print(f"Resp: {response} Acc: {round(count/all, 4)} {count}/{all}", flush=True)
            print('=====================')