import os 
import json
import argparse
from torch.utils.data import Dataset
import time, torch
from eval.utils import *

class VidHalDataset(Dataset):
    def __init__(self, data_path, video_root) -> None:
        super().__init__()

        with open(data_path, "r") as f:
            self.examples = json.load(f)
        
        self.video_root = video_root
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        video_name, captions, aspect = example["video"], example["captions"], example["aspect"]
        video_path = os.path.join(self.video_root, f"{video_name}.mp4")
        return {
            "video_name": video_name, "video_path" : video_path, "captions" : captions, "aspect" : aspect
        }


def format_options_prompt(captions, video_id=None, option_to_rank=None):
    """
    Generates the sub-prompt containing line-break separated [option : caption] to be displayed to the model
    """
    assert option_to_rank is not None or video_id is not None # Either video ID provided to use pre-defined ordering, or custom option ordering must be provided
    if option_to_rank is None:
        option_to_rank = self.option_display_order[video_id]

    options_prompt = "\n".join([f"{option}. {captions[rank]}" for option, rank in option_to_rank.items()])
    for option, rank in option_to_rank.items():
        if rank == '1':
            gt = f"{option}."
    return options_prompt, gt, captions['1']

    
#system_prompt_instruction = "You are provided with a video and a set of several captions. Your task is to watch the video provided carefully, and " \
system_prompt_instruction = "which caption describes the video most accurately? answer with the option whose caption that best describes the video\n" #"select the caption that best describes the video.\n" 
no_cot_instruction = "Provide your answer only as a single letter representing the option whose caption that best describes the video, without any explanation.\n"
# main_prompt_instruction = "Watch the video provided, and choose the option whose caption describes the video most accurately."
cot_instruction = '''\nFirst outline your reasoning step-by-step by referencing to the frames, and then clearly state your final answer as a single letter representing the option.\n\n'''
            
def process_response(response):
    """
    Parses the generated response to extract only the selected option.
    """
    last_option = list(string.ascii_uppercase)[self.num_captions - 1]
    match = re.search(fr"\b[a-{last_option.lower()}A-{last_option}]\b", response)
    match = match.group(0).upper().strip(";:., ") if match else None

    return match if match else response # If no match, keep original response in case model replies with caption instead of option


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='OpenGVLab/InternVL2_5-1B')
    parser.add_argument('--annotation-path', type=str, default='/vast/sg7457/video_datasets/VidHal/annotations.json')
    parser.add_argument('--option-path', type=str, default='/vast/sg7457/video_datasets/VidHal/options.json')
    parser.add_argument('--video-path', type=str, default='/vast/sg7457/video_datasets/VidHal/videos')
    parser.add_argument('--cot', action='store_true')
    torch.manual_seed(42)
    args = parser.parse_args()
    if 'intern' in args.model_path.lower(): 
        model, tokenizer, generation_config = load_model(args.model_path, max_new_tokens=2048)
    elif 'llava' in args.model_path.lower(): 
        model, processor = load_llava_next_model(args.model_path)
    else:
        model, processor = load_qwen_model(args.model_path)
    
    dataset = VidHalDataset(args.annotation_path, args.video_path)
    if args.option_path:
        with open(args.option_path, "r") as f:
            option_display_order = json.load(f)
    else:
        option_display_order = None
        
    start_t = time.time()
    count, all = 0, 0
    for sample_idx, sample in enumerate(dataset):
        try:
            prompt, gt, gt_text = format_options_prompt(captions=sample['captions'], video_id=sample['video_name'], 
                                        option_to_rank=option_display_order[sample['video_name']])
            
            if args.cot:
                question = add_cot_to_question(system_prompt_instruction + prompt)
            else:
                question = system_prompt_instruction + prompt
                
            if 'intern' in args.model_path.lower(): 
                pixel_values, num_patches_list = load_video(sample['video_path'], bound=None, input_size=448, max_num=1, num_segments=30)
                response = generate_response_for_video(pixel_values, num_patches_list, question, model, tokenizer, generation_config)
            
            elif 'llava' in args.model_path.lower():
                response = get_llava_next_response(sample['video_path'], question, model, processor)
            
            else:
                fps = 1.0
                # if '72B' in args.model_path:
                #     fps=0.5
                response = get_qwen_vl_response(sample['video_path'], question, model, processor, fps=fps)
            
            interval = round((time.time() - start_t)/60, 2)
            
            count += (gt in response)# or gt_text.lower() in response.lower())

            all += 1
            # print(sample['realpath'])
            print(f"{sample_idx}/{len(dataset)}, {sample['video_path']}, Time: {interval} Q: {question}")
            print(f"GT:{gt} {gt_text}, Resp: {response} Acc: {round(count/all, 4)} {count}/{all}", flush=True)
        except:
            pass