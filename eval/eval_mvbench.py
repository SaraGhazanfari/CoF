import argparse, json, os
from eval.utils import *
from metrics.cider_score import Cider
import time, torch

root_dir = '/vast/sg7457/video_datasets/MVBench/complete_json'

def get_question_and_answer(question, options, answer, has_cot):
    
    option_alph_list = ['(A)',  '(B)',  '(C)', '(D)']
    # question= '''You have been provided with several video frames that show an event occurring over time. 
    # Carefully analyze each frame, describing your observations clearly. Provide step-by-step reasoning by refering to the frames and details in these frames. 
    # After completing your reasoning, separately state your final answer to the following question\n'''+  "Question: " + question
    
    question += '\n'
    for idx, option in enumerate(options):
        question += f'{option_alph_list[idx]} {option}\n'
        if option == answer:
            answer = option_alph_list[idx]
    if has_cot:
        question = add_cot_to_question(question)
    return question, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/InternVL2_5-1B-TimeIT-v2')
    parser.add_argument('--task', type=str)
    parser.add_argument('--cot', action='store_true')
    torch.manual_seed(42)
    args = parser.parse_args()

    model, tokenizer, generation_config = load_model(args.model_path, max_new_tokens=2048)
    test_data = list()

    start_t = time.time()
    
    for json_file in os.listdir(root_dir):
        count, all = 0, 0

        try:
            with open(os.path.join(root_dir, json_file)) as file:
                print(json_file)
                test_data = json.load(file)         
            for sample_idx, sample in enumerate(test_data):
                 

                pixel_values, num_patches_list = load_video(sample['realpath'], bound=None, input_size=448, max_num=1, num_segments=30)
                question, answer = get_question_and_answer(sample['question'], sample['candidates'], sample['answer'], args.cot)
                response = generate_response_for_video(pixel_values, num_patches_list, 
                                                    question, model, tokenizer, generation_config)

                interval = round((time.time() - start_t)/60, 2)
                count += (answer in response)
                all += 1
                # print(sample['realpath'])
                print(f'{sample_idx}, Time: {interval}, Q: {question}')
                print(f'GT:{answer}, Resp: {response} Acc: {round(count/all, 4)} {count}/{all}', flush=True)
            print(f'{json_file}, Acc: {round(count/all, 4)} {count}/{all}', flush=True)
            
        except Exception as e:
            print(e)
            print(f'{json_file} has issue!!!')
        
     
     
     
            
            
            
            
# root_dir = '/vast/sg7457/video_datasets/MVBench'
# json_dir = 'json'
# complete_json_dir = 'complete_json'
# video_dir = 'video'
# video_categories = ['clevrer/video_validation',
#                     'Moments_in_Time_Raw/videos',
#                     'scene_qa/video',
#                     'FunQA_test/test',
#                     'sta/sta_video',
#                     'vlnqa',
#                     'data0613/clevrer',
#                     'data0613/star',
#                     'star/Charades_v1_480',
#                     'tvqa/frames_fps3_hq',
#                     'ssv2_video',
#                     'perception/videos']



# count = 0
# for idx, json_file in enumerate(os.listdir(os.path.join(root_dir, json_dir))):
#     # if idx <= 20:
#     #     continue
#     if json_file.endswith('.json'):
#         with open(os.path.join(root_dir, json_dir, json_file), 'r') as f:
#             count = 0
#             data = json.load(f)
#             save_data = list()
#             for sample in data:
#                 found = False
#                 for video_cat in video_categories:
#                     video_path = os.path.join(root_dir, video_dir, video_cat, sample['video'])
#                     if os.path.exists(video_path):
#                         sample['realpath'] = video_path
#                         found=True
#                         save_data.append(sample)
#                         break
#                 if not found:
#                     # print(sample['video'])
#                     count += 1
#         if len(save_data) > 0: 
#             with open(os.path.join(root_dir, complete_json_dir, json_file), 'w') as f:
#                 json.dump(save_data, f)
                      
#         print(json_file, count, len(data))
