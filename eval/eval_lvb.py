import argparse, json
import time, os
from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
from eval.utils import *
import json

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        interleaved_list.append(frame)
        
    return interleaved_list
    
class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 duration_group,
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            data = json.load(f)
            
        self.data = [sample for sample in data if sample['duration_group']==duration_group]
        self.max_num_frames = max_num_frames
        
        
        
    def __getitem__(self, index):
        di = self.data[index]
        
        frames, frame_timestamps = load_video(os.path.join(self.data_path, "videos", di["video_path"]), di["duration"], max_num_frames=self.max_num_frames)
        
            
        with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
            subtitles = json.load(f)
        inputs = []
        if self.insert_text:
            inputs = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])
        else:
            inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        question = ''
        question += f"**Question:** {di["question"]}\n"  
        question += '\n'.join([". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])])
        question += "\nAnswer with the option's letter from the given choices directly.\n"
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        return {"video": os.path.join(self.data_path, "videos", di["video_path"]),
                "inputs": inputs, 
                "question": question,
                "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), 
                "id": di["id"]}
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]
 
def interleaved_eval_model():
    count, all = 0, 0
    for idx in range(len(db)):
        
        frame_idx = 0
        img_list = list()
        desc = dict()
        question = db[idx]['question']
        for input_ in db[idx]['inputs']:
            if isinstance(input_, str):
               desc[frame_idx-1] = input_
            else:
                img_list.append(input_)
                frame_idx += 1
        
        
        pixel_values, num_patches_list = load_frames(img_list)
        response = generate_response_for_subtitled_video(pixel_values, num_patches_list, 
                                            question, model, tokenizer, 
                                            generation_config,subtitle=desc)

        if not args.cot:
            count += (db[idx]['correct_choice'] in response)
            
        else:
            if db[idx]['correct_choice']+'.' in response:
                count +=1
            elif response.split('\n') == 1 and db[idx]['correct_choice'] in response:
                count += 1
            elif 'Answer:' in response and db[idx]['correct_choice'] in response.split('Answer:')[1]:
                count +=1 

        all += 1
        print(question, flush=True)
        print(f'{idx}/{len(db)} gt: {db[idx]['correct_choice']}, Resp: {response} Acc: {round(count/all, 4)} {count}/{all}', flush=True)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/InternVL2_5-1B-TimeIT-v1-5')
    parser.add_argument('--task', type=str)
    parser.add_argument('--duration', type=int)
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--no-sub', action='store_true')
    
    args = parser.parse_args()
    
    model, tokenizer, generation_config = load_model(args.model_path)
    db = LongVideoBenchDataset("/vast/sg7457/video_datasets/LongVideoBench", 
                               "/vast/sg7457/video_datasets/LongVideoBench/lvb_val.json",
                               max_num_frames=30,
                               duration_group=args.duration)
    
    count, all = 0, 0
    for idx in range(len(db)):
        print(db[idx]['video'])
        frame_idx = 0
        img_list = list()
        desc = ''
        question = db[idx]['question']
        for input_ in db[idx]['inputs']:
            if isinstance(input_, str):
                desc += f'Frame {frame_idx}: {input_}\n'
            else:
                img_list.append(input_)
                frame_idx += 1
        if args.no_sub:
            if args.cot:
                instruction = add_cot_to_question(question)
            else:
                instruction = question
        else:
            if args.cot:
                instruction = add_cot_with_subtitle_to_question(question, desc)
            else:
                instruction = add_subtitles_to_question(question, desc)
        
        pixel_values, num_patches_list = load_frames(img_list)
        response = generate_response_for_video(pixel_values, num_patches_list, 
                                            instruction, model, tokenizer, generation_config)

        if not args.cot:
            count += (db[idx]['correct_choice'] in response)
            
        else:
            if db[idx]['correct_choice']+'.' in response:
                count +=1
            elif (response.split('\n') == 1 or '\n' not in response) and db[idx]['correct_choice'] in response:
                count += 1
            elif 'Answer:' in response and db[idx]['correct_choice'] in response.split('Answer:')[1]:
                count +=1 

        all += 1
        print(instruction, flush=True)
        print(f'{idx}/{len(db)} gt: {db[idx]['correct_choice']}, Resp: {response} Acc: {round(count/all, 4)} {count}/{all}', flush=True)
    