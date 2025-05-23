import argparse, json, os
from eval.utils import *
import time, torch
from datasets import load_dataset




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--part', type=int, choices=[1,2,3]) # 1-> t<900, 2-> 900<=t<1800, 3-> t>=1800
    parser.add_argument('--cot', action='store_true')
    torch.manual_seed(42)   
    args = parser.parse_args()
    data_root = '/vast/sg7457/video_datasets/TimeIT/next_qa/'
    ann_file = f'annotations/{args.split}.csv'
    if 'intern' in args.model_path.lower(): 
        model, tokenizer, generation_config = load_model(args.model_path)
    else:
        model, processor = load_qwen_model(args.model_path)
        
    import pandas as pd

    df = pd.read_csv(os.path.join(data_root, ann_file))
    options = ['A.', 'B.', 'C.', 'D.', 'E.']
    count, all = 0, 0
    # Show the first few rows
    for index, sample in df.iterrows():
        if args.part == 1 and sample['frame_count'] < 900 or \
            args.part == 2 and 900<=sample['frame_count']<1800 or \
            args.part == 3 and 1800<=sample['frame_count']:
            question = sample['question']+'\n'
            for opt_idx in range(len(options)):
                question += options[opt_idx] + sample[f'a{opt_idx}']+'\n'
            
            if args.cot:
                question = add_cot_to_question(question, options='"A.", "B.", "C.", "D.", "E."')
        
            video = os.path.join(data_root, f"videos/{sample['video']}.mp4")
            if 'intern' in args.model_path.lower(): 
                pixel_values_v2, num_patches_list_v2= load_video(video, bound=None, 
                                                                        input_size=448, 
                                                                        max_num=1, 
                                                                        num_segments=30)
                
                response = generate_response_for_video(pixel_values_v2, num_patches_list_v2, 
                                                question, model, tokenizer, generation_config)
            else:
                print('fps=1.0')
                try:
                    response = get_qwen_vl_response(video, question, model, processor, fps=1.0)
                except:
                    try:
                        print('fps=1.0')
                        response = get_qwen_vl_response(video, question, model, processor, fps=0.5)
                    except:
                        print('fps=1.0')
                        response = get_qwen_vl_response(video, question, model, processor, fps=0.25)
                
            count += (options[sample['answer']] in response)
            all += 1
            print(f"gt: {options[sample['answer']]}, resp: {response} Acc: {round(count/all, 4)} ({count}/{all})", 
                  flush=True)
        
         
     
    print(f"Finished gt: {options[sample['answer']]}, resp: {response} Acc: {round(count/all, 4)} ({count}/{all})",
          flush=True)