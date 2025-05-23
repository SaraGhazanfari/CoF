import argparse
from datasets import load_dataset
import os, time, torch
from eval.utils import *

# {'object_counting': 565,
#  'object_rel_distance': 710,
#  'obj_appearance_order': 618,

#  'object_size_estimation': 953,
#  'room_size_estimation': 288,
#  'object_abs_distance': 834,
#  'object_rel_direction_hard': 373,
#  'object_rel_direction_medium': 378,
#  'object_rel_direction_easy': 217,
#  'route_planning': 194}

num_map = {
    "single": 1, "multiple": 5,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30, "thirty-one": 31, "thirty-two": 32,
    "thirty-three": 33, "thirty-four": 34, "thirty-five": 35,
    "thirty-six": 36, "thirty-seven": 37, "thirty-eight": 38,
    "thirty-nine": 39, "forty": 40, "forty-one": 41, "forty-two": 42,
    "forty-three": 43, "forty-four": 44, "forty-five": 45,
    "forty-six": 46, "forty-seven": 47, "forty-eight": 48,
    "forty-nine": 49, "fifty": 50, "fifty-one": 51, "fifty-two": 52,
    "fifty-three": 53, "fifty-four": 54, "fifty-five": 55,
    "fifty-six": 56, "fifty-seven": 57, "fifty-eight": 58,
    "fifty-nine": 59, "sixty": 60, "sixty-one": 61, "sixty-two": 62,
    "sixty-three": 63, "sixty-four": 64, "sixty-five": 65,
    "sixty-six": 66, "sixty-seven": 67, "sixty-eight": 68,
    "sixty-nine": 69, "seventy": 70, "seventy-one": 71, "seventy-two": 72,
    "seventy-three": 73, "seventy-four": 74, "seventy-five": 75,
    "seventy-six": 76, "seventy-seven": 77, "seventy-eight": 78,
    "seventy-nine": 79, "eighty": 80, "eighty-one": 81, "eighty-two": 82,
    "eighty-three": 83, "eighty-four": 84, "eighty-five": 85,
    "eighty-six": 86, "eighty-seven": 87, "eighty-eight": 88,
    "eighty-nine": 89, "ninety": 90, "ninety-one": 91, "ninety-two": 92,
    "ninety-three": 93, "ninety-four": 94, "ninety-five": 95,
    "ninety-six": 96, "ninety-seven": 97, "ninety-eight": 98,
    "ninety-nine": 99, "one hundred": 100
}
def get_response(img_root, data, question, model_path):
    video_path = os.path.join(img_root, f"{data['dataset']}/{data['scene_name']}.mp4")
    print(video_path, flush=True)
    if 'intern' in model_path:
        pixel_values, num_patches_list = load_video(video_path, num_segments=30)
        response = generate_response_for_video(pixel_values, num_patches_list, question, model, tokenizer, generation_config)
    else:
        fps = 0.5
        #0.25
        # if '72B' in model_path:
        #     fps=0.1
        response = get_qwen_vl_response(video_path, question, model, processor, fps=fps)
    return response

def mean_acc(answer_list):
    mac_list = list()       
    for answer in answer_list:
        try:
            gt = float(answer['gt'])
            pred = float(answer['pred'])
            if pred < 0:
                pred = pred * (-1)
            mac_list.append(mean_relative_accuracy(pred, gt))
        except Exception as e:
            print(e)
            print(answer)    
        
    
    print('Mean Accuracy:', round(sum(mac_list) / len(mac_list), 4))
    print('Total:', len(mac_list))  
    return round(sum(mac_list) / len(mac_list), 4)
    
def extract_count_from_response(response, args):
    # numbers = extract_num_from_response(response, args)
    # if numbers:
    #     return numbers
    if 'answer:' in response:
        response = response.split('answer:')[1].split('\n')[0]
    elif '\n' in response:
        response_list = response.split('\n')
        response = [resp for resp in response_list if resp != ''][-1]
    for key, value in reversed(num_map.items()):
        if key in response:
            return str(value)


def extract_num_from_response(response, args):
    if 'answer:' in response:
        response = response.split('answer:')[1].split('\n')[0]
    elif '\n' in response:
        response_list = response.split('\n')
        response = [resp for resp in response_list if resp != ''][-1]
    
    import re
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    parsed_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    if len(parsed_numbers) == 0:
        return None
    
    return str(parsed_numbers[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--cot', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    if 'intern' in args.model_path.lower():
        model, tokenizer, generation_config = load_model(args.model_path)
    else:
        model, processor = load_qwen_model(args.model_path)
        
    reverse_num_map = {str(v): k for k, v in num_map.items()}

    ds = load_dataset("nyu-visionx/VSI-Bench", cache_dir='/vast/sg7457/video_datasets')
    img_root = '/vast/sg7457/video_datasets/VSI-Bench'
    count, all = 0, 0
    start_time = time.time()

    count = 0
    data_dict = dict()
    answer_list = list()
    response, question = '', ''
    prev_all = 0
    for data in ds['test']:
        try:
            gt_idx_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            
            if args.task == 'obj_appearance_order' and data['question_type'] == 'obj_appearance_order':
                question = data['question'] +'\n'+'\n'.join(data['options']) 
                if args.cot:
                    question = add_cot_to_question(question, options='"A.", "B.", "C.", "D."') 
                response = get_response(img_root, data, question, args.model_path)
                gt = data['options'][gt_idx_mapping[data['ground_truth']]].replace(f"{data['ground_truth']}.",'').strip()
                count += (f"{data['ground_truth']}." in response or gt in response)
                all += 1 


            elif args.task == 'object_rel_distance' and data['question_type'] == 'object_rel_distance':

                question = data['question'] +'\n'+'\n'.join(data['options']) 
                if args.cot:
                    question = add_cot_to_question(question, options='"A.", "B.", "C.", "D."')
                else:
                    question += '\nselect the correct choice.'
                response = get_response(img_root, data, question, args.model_path)
                count += (f"{data['ground_truth']}." in response)
                all += 1 
                
            elif args.task == 'object_rel_direction' and data['question_type'] in ['object_rel_direction_hard', 
                                                                                'object_rel_direction_medium',
                                                                                'object_rel_direction_easy']:
                question = data['question'] +'\n'+'\n'.join(data['options']) 
                if args.cot:
                    question = add_cot_to_question(question, options='"A.", "B.", "C.", "D."') 
                response = get_response(img_root, data, question, args.model_path)
                gt = data['options'][gt_idx_mapping[data['ground_truth']]].replace(f"{data['ground_truth']}.",'').strip()
                count += (f"{data['ground_truth']}." in response or gt in response)  
                all += 1 
                
            elif args.task == 'route_planning' and data['question_type'] == 'route_planning':
                
                question = data['question'] +'\n'+'\n'.join(data['options']) 
                if args.cot:
                    question = add_cot_to_question(question, options='"A.", "B.", "C.", "D."') 
                response = get_response(img_root, data, question, args.model_path)
                gt = data['options'][gt_idx_mapping[data['ground_truth']]].replace(f"{data['ground_truth']}.",'').strip()
                count += (f"{data['ground_truth']}." in response or gt in response) 
                all += 1 
                
            elif args.task == 'object_counting' and data['question_type'] == 'object_counting':   
                question = data['question'] #+ '\ngive the answer in numerical form'
                if args.cot:
                    question = add_cot_to_question(question)
                response = get_response(img_root, data, question, args.model_path).lower()
                print(response)
                pred = extract_count_from_response(response, args)
                count += (pred in data['ground_truth'])    
                all += 1 
                answer_list.append({
                    'pred':pred,
                    'gt': data['ground_truth']
                })
                
            elif args.task == 'object_size_estimation' and data['question_type'] == 'object_size_estimation':  
                question = data['question'] #+ '\ngive the answer in numerical form'
                if args.cot:
                    question = add_cot_to_question(question)
                response = get_response(img_root, data, question, args.model_path).lower()
                pred = extract_num_from_response(response, args)
                answer_list.append({
                    'pred':pred,
                    'gt': data['ground_truth']
                })
                count += (pred in data['ground_truth'])   
                all += 1 
                
            elif args.task == 'room_size_estimation' and data['question_type'] == 'room_size_estimation':
                question = data['question'] #+ '\ngive the answer in numerical form'
                if args.cot:
                    question = add_cot_to_question(question)
                response = get_response(img_root, data, question, args.model_path).lower()
                pred = extract_num_from_response(response, args)
                answer_list.append({
                    'pred':pred,
                    'gt': data['ground_truth']
                })
                count += (pred in data['ground_truth'])   
                all += 1 
                
            elif args.task == 'object_abs_distance' and data['question_type'] == 'object_abs_distance':
                
                question = data['question'] #+ '\ngive the answer in numerical form'
                if args.cot:
                    question = add_cot_to_question(question)
                response = get_response(img_root, data, question, args.model_path).lower()
                pred = extract_num_from_response(response, args) 
                answer_list.append({
                    'pred':pred,
                    'gt': data['ground_truth']
                })
                count += (pred in data['ground_truth'])   
                all += 1 
            
            
            interval_t = round((time.time() - start_time)/60, 2)
            if all != 0 and all != prev_all:
                prev_all = all
                if len(answer_list) > 0:
                    print(f"Time:{interval_t} GT: {data['ground_truth']} Pred:{pred}, Acc: {round(count/all, 4)} ({count}/{all})", flush=True)  
                else:
                    print(f"Time:{interval_t} GT: {data['ground_truth']}, Acc: {round(count/all, 4)} ({count}/{all})", flush=True)
                print(f"User: {question}\nAssistant: {response}", flush=True)
        except Exception as e:
            print(e)
            print(response)
    
           
    if len(answer_list) > 0:
        print(f"{args.task}, MAcc: {mean_acc(answer_list)} Acc: {round(count/all, 4)} ({count}/{all})", flush=True)
    else:
        print(f"{args.task} Acc: {round(count/all, 4)} ({count}/{all})", flush=True)