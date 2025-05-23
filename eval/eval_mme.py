import argparse, json, os
from eval.utils import *
from metrics.cider_score import Cider
import time, torch
from datasets import load_dataset


root_dir = '/vast/sg7457/video_datasets/MVBench/complete_json'
def get_question_and_answer(question, options, answer):
    option_alph_list = ['(A)',  '(B)',  '(C)', '(D)']
    question += '\n'
    for idx, option in enumerate(options):
        question += f'{option_alph_list[idx]} {option}\n'
        if option == answer:
            answer = option_alph_list[idx]
    return question, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/vs/InternVL2_5-1B-VS-v2-266k')
    parser.add_argument('--task', type=str)
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--duration', type=str, default=None)
    torch.manual_seed(42)
    args = parser.parse_args()
    video_root_dir = '/vast/sg7457/video_datasets/Video-MME/data-compressed'
    video_id_dict = dict()
    ds = load_dataset("lmms-lab/Video-MME")
    video_list = os.listdir(video_root_dir)
    answer_list = list()
    for sample_idx, sample in enumerate(ds['test']):
        
        for video_name in video_list:
            if sample['videoID'] in video_name:
                video_id_dict[sample['videoID']] = os.path.join(video_root_dir, video_name)
             
    if 'intern' in args.model_path.lower(): 
        model, tokenizer, generation_config = load_model(args.model_path)
        
    elif 'llava' in args.model_path.lower(): 
        tokenizer, model, image_processor = load_llava_next_model(args.model_path)
        
    else:
        model, processor = load_qwen_model(args.model_path)
        
    test_data = list()

    start_t = time.time()
    stat_dict = dict()
    count_dict = dict()
    fps_dict = {'short':0.5, 'medium':0.1, 'long':0.02}
    
    for sample_idx, sample in enumerate(ds['test']):
        if args.duration and sample['duration']!=args.duration:
            continue
        video = video_id_dict[sample['videoID']]
        
        answer = f"{sample['answer']}."
        question = sample['question'] + '\n' + '\n'.join(sample['options'])
        for opt in sample['options']:
            if opt.startswith(answer):
                answer_text = opt.replace(answer, '').strip()
                if answer_text.endswith('.'):
                    answer_text = answer_text[:-1]
  
        if args.cot:
            question = add_cot_to_question(question, options='"A.", "B.", "C.", "D."')
        else:
            question = question

        if 'intern' in args.model_path.lower():
            pixel_values_v2, num_patches_list_v2 = load_video(video, bound=None, #[frame_num, video_len], 
                                                                input_size=448, 
                                                                max_num=1, 
                                                                num_segments=30)

            
            response = generate_response_for_video(pixel_values_v2, num_patches_list_v2, 
                                                question, model, tokenizer, generation_config)
        elif 'llava' in args.model_path.lower():
            response = get_llava_next_response(video, question, tokenizer, model, image_processor)
        else:
            response = get_qwen_vl_response(video, question,  model, processor, 
                                            fps=fps_dict[sample['duration']])
                
        # Logging
        interval = round((time.time() - start_t)/60, 2)
        
        if sample['duration'] not in count_dict:
            count_dict[sample['duration']] = [0, 0]
            
        count_dict[sample['duration']][0] += (answer in response)
        answer_list.append({'status': (answer in response), 'resp': response})
        count_dict[sample['duration']][1] += 1
        print(f'video: {video}, Q: {question}')
        print(f"{sample_idx}/{len(ds['test'])}, Time: {interval}, GT:{answer}, Acc: {round(count_dict[sample['duration']][0]/count_dict[sample['duration']][1],4)} ({count_dict[sample['duration']][0]}/{count_dict[sample['duration']][1]}) Resp: {response}", flush=True)
  
    for key, value in count_dict.items():
        print(f'{key}, Acc: {round(value[0]/value[1],4)} ({value[0]}/{value[1]})', flush=True)  

    with open(f'{args.model_path.split('/')[-1]}_mme_resp.json', 'w') as file:
        json.dump(answer_list, file)
            
     
