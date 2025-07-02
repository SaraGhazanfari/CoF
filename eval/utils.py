

import torch, math

from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import cv2
import random


def split_model(model_name):
    
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def load_model(model_path, max_new_tokens=1024):
    from transformers import AutoModel, AutoTokenizer
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
    config = InternVLChatConfig.from_pretrained(model_path)
    model = InternVLChatModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config, cache_dir='./').eval().cuda()
    # name_only = model_path.split('/')[-1].split('B')[0]+'B'
    # device_map = split_model(name_only)
    # model = AutoModel.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     use_flash_attn=True,
    #     trust_remote_code=True,
    #     device_map=device_map,
    #     cache_dir='./').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, cache_dir='./')

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    
    return model, tokenizer, generation_config


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video_to_frames(video_path, bound=None, num_segments=30):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    img_list = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img_list.append(img)
        
    return img_list


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, return_imgs=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    img_list = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img_list.append(img)
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    if return_imgs:
        return pixel_values, num_patches_list, img_list, max_frame, fps
    return pixel_values, num_patches_list


def generate_response_for_video(pixel_values, num_patches_list, question, model, tokenizer, generation_config, question_prefix=''):
    # pixel_values, num_patches_list = load_video(video_path, num_segments=30, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    instruction = question_prefix + video_prefix + question
    with torch.no_grad():
        response, history = model.chat(tokenizer, pixel_values, instruction, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        
    if 'Question:' in response and '**Reasoning:**' in response:
        response = response.split('**Reasoning:**')[1]
    return response
    
def generate_response_for_subtitled_video(pixel_values, num_patches_list, question, model, tokenizer, generation_config, subtitle):
    # pixel_values, num_patches_list = load_video(video_path, num_segments=30, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    frame_list = list()
    for i in range(len(num_patches_list)):
        if i in subtitle:
            frame_list.append(f'Frame{i+1}: <image>, {subtitle[i]}\n')
        else:
            frame_list.append(f'Frame{i+1}: <image>\n')
    video_prefix = ''.join(frame_list)
    instruction = video_prefix + question

    response, history = model.chat(tokenizer, pixel_values, instruction, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    
    return response

def load_frames(frames, input_size=448, max_num=1):

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    
    for img in frames:
        # img = Image.open(frame).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def get_duration(video_path):
    
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps
    cap.release()
    return duration

def random_window_with_k(k, X, window_size):
    start = max(0, k - window_size + 1)

    start = random.randint(start, k)
    return [start, start+window_size]

def add_cot_v2_to_question(question):
    question = "You are given a video and a related question. Follow these steps to reason and answer:\n" \
    "1. Identify the key frames in the video that are relevant to the question.\n" \
    "2. Use those frames to logically break down your reasoning toward an answer.\n" \
    "3. Keep your reasoning concise and clear.\n" \
    "4. Based on your reasoning, provide a direct and accurate answer.\n\n" \
    "Question: " + question
    question += '\n **Reasoning:** '
    
    return question

def add_cot_to_question(question, options=None, binary=False):
    # if binary:
    #     question = "Given a video and a question, Start reasoning step-by-step like this:\n" \
    #                 "Point out key frames from the video relevant to the question.\n" \
    #                 "Break down the reasoning from those frames to the answer.\n" \
    #                 'Conclude your reasoning to the answer by selecting the "Yes" or "No".\n\nQuestion: ' + question     
    # elif options:
    #     question = "Given a video and a question, Start reasoning step-by-step like this:\n" \
    #                 "Point out key frames from the video relevant to the question.\n" \
    #                 "Break down the reasoning from those frames to the answer.\n" \
    #                 f'Conclude your reasoning to the answer by selecting the correct option. ({options})\n\nQuestion: ' + question     
                    
    # else:
    question = "Given a video and a question, Start reasoning step-by-step like this:\n" \
                "Point out key frames from the video relevant to the question.\n" \
                "Break down the reasoning from those frames to the answer.\n" \
                f'Conclude your reasoning to the answer.\n\nQuestion: ' + question 
    question += '\n **Reasoning:** '
    
    return question

def add_cot_with_subtitle_to_question(question, desc):
    question = "Given a video , its frame-wise subtitles and a question, Start reasoning step-by-step like this:\n" \
                "Point out key frames from the video relevant to the question.\n" \
                "Break down the reasoning from those frames and their subtitles to the answer.\n" \
                "Answer clearly.\n\n"\
                f"**frame-wise subtitles:**\n{desc}\n\n{question}"      
    question += '\n Reasoning: '
    
    return question

def add_subtitles_to_question(question, desc):
    question = "Given a video, its frame-wise subtitles and a question, answer the question\n\n" \
                f"**frame-wise subtitles:**\n{desc}\n\n" \
                f"{question}" 
    
    return question


def add_frame_to_question(question):
    question = "Given a video and a question, \n"\
    "Find the frame_number of the video that is most relevant to the question:\n\nQuestion: " + question     
    question += '\n **Frame number:** '
    
    return question

def add_range_to_question(question):
    question = "Given a video and a question, \n"\
    "Point out key frames from the video relevant to the question.\n" \
    "Give the frame number range as [x, y].\n\nQuestion: " + question     
    question += '\n **Frame number range:** '
    
    return question


def video_processor(video_path, num_segments, ratio=1, bound=None):
    import av
    container = av.open(video_path)
    fps = container.streams.video[0].average_rate
    start_pts = int(bound[0] * fps)
    end_pts = int(bound[1] * fps)
    ratio = int(ratio * fps)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i < start_pts:
            continue
        if i >= end_pts:
            break
        if (i - start_pts) % ratio == 0:
            img = frame.to_image()  # Converts to PIL Image
            frames.append(img)
    return load_frames(frames), frames


def load_qwen_model(model_path):
    # from transformers import Qwen2_VLForConditionalGeneration, AutoProcessor
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", cache_dir='./'
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir='./')
    return model, processor

def load_llava_next_model(model_path):
    from transformers import BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
    import torch


    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, torch_dtype="float16", device_map='auto', cache_dir='./')
    processor = LlavaOnevisionProcessor.from_pretrained(model_path, cache_dir='./')
    processor.tokenizer.padding_side = "left" # set to 'left' for generation and 'right' for training (default in 'right')
    
    return model, processor


def get_qwen_vl_response(video, prompt, model, processor, fps=1.0):
    from qwen_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
                    # "max_pixels": 401408,
                    # "min_pixels": 200704,
                    "max_pixels": 360 * 420,
                    "video-min-frames": 30,
                    "video-max-frames": 30,
                    "fps":fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        # min_video_frames=30,
        # max_video_frames=30,
        return_tensors="pt",
        # fps=fps
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def get_llava_next_response(video, question, model, processor):
    import av
    import numpy as np

    def read_video_pyav(video_path, max_num_frames=30):
        '''
        Decode the video with PyAV decoder.

        Args:
            container (av.container.input.InputContainer): PyAV container.
            indices (List[int]): List of frame indices to decode.

        Returns:
            np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        import av
        container = av.open(video_path)
        fps = container.streams.video[0].average_rate
        ratio=1
        if container.streams.video[0].frames > 30:
            ratio = round(container.streams.video[0].frames / 30)
        
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % ratio == 0:
                img = frame.to_image()  # Converts to PIL Image
                frames.append(img)
        print(len(frames))                
        return frames
    
        # indices = range(len)
        # frames = []
        # container.seek(0)
        # start_index = indices[0]
        # end_index = indices[-1]
        # for i, frame in enumerate(container.decode(video=0)):
        #     if i > end_index:
        #         break
        #     if i >= start_index and i in indices:
        #         frames.append(frame)
        # return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    conversation_2 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": question+"\n"},
              {"type": "video"},
              ],
        },
    ]
    prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
    inputs = processor(text=[prompt_2], videos=[read_video_pyav(video)], padding=True, return_tensors="pt").to(model.device, torch.float16)
    generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0].split('\n')[-1]
    return generated_text

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


# mac_list = list()
# with open('v0_ans.txt') as file:
#     for line in file.readlines():
#         gt = float(line.split('GT: ')[1].split(' ')[0].strip())
#         pred = float(line.split('Pred:')[1].split(',')[0].strip())
#         print(gt, pred)
#         mac_list.append(mean_relative_accuracy(pred, gt))
        
# print('Mean Accuracy:', round(sum(mac_list) / len(mac_list), 4))
