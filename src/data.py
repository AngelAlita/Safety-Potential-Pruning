import numpy as np
import random
import torch
import transformers 
from transformers import AutoProcessor
from PIL import Image
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


# Wrapper for tokenized input IDs

def get_hod(args, nsamples, seed, image_folder, processor):
    # Load datasets
    data = load_dataset('json', data_files='./data/HOD/HOD.json', split='train')
    random.seed(seed)
    norm_loader = []
    safety_loader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(data))


        img_path = image_folder + '/' + data['data'][idx]['image_path']
        
        image = Image.open(img_path).resize((1024,1024))
        if "llava-v1.6" in args.model:
            norm_messages = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": data['data'][idx]['question']},
                    {"type": "image"},
                    ],
                },
            ]
            safety_messages = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": data['data'][idx]['rephrased_question']},
                    {"type": "image"},
                    ],
                },
            ]
            norm_prompt = processor.apply_chat_template(norm_messages, add_generation_prompt=True)
            safety_prompt = processor.apply_chat_template(safety_messages, add_generation_prompt=True)
            norm_inputs = processor(images=image, text=norm_prompt, return_tensors="pt")
            safety_inputs = processor(images=image, text=safety_prompt, return_tensors="pt")
        elif "Qwen" in args.model:
            norm_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data['data'][idx]['question']},
                    ],
                }
            ]
            safety_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data['data'][idx]['rephrased_question']},
                    ],
                }
            ]
            # Preparation for inference
            norm_text = processor.apply_chat_template(norm_messages, tokenize=False, add_generation_prompt=True)
            safety_text = processor.apply_chat_template(safety_messages, tokenize=False, add_generation_prompt=True)
            norm_inputs = processor(
                text=[norm_text], images=[image], padding=True, return_tensors="pt"
            )
            safety_inputs = processor(text=[safety_text], images=[image], padding=True, return_tensors="pt")
        elif "blip" in args.model:
            norm_inputs = processor(images=image, text=data['data'][idx]['question'], return_tensors="pt")
            safety_inputs = processor(images=image, text=data['data'][idx]['rephrased_question'], return_tensors="pt")

        norm_loader.append(norm_inputs)
        safety_loader.append(safety_inputs)
    return norm_loader, safety_loader
        




def get_mmsafetybench(args, nsamples, seed, image_folder, processor):
    # Load datasets
    data = load_dataset('json', data_files='./data/MM-SafetyBench/mmsafetybench_train.json', split='train')
    data = data['data'][0]
    random.seed(seed)
    
    norm_loader = []
    safety_loader = []
    seq = random.sample(range(len(data)), nsamples)
    for _ in range(args.nsamples):
        idx = seq[_]
        image = Image.open(image_folder + '/' + data[idx]['scenario'] + '/' + data[idx]['image_path']).convert('RGB')
        norm_messages= [ {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data[idx]['question']},
                    ]
                }]
        
        safety_messages= [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": data[idx]['rephrased_question']},
                ],
            }]


        # Preparation for inference
        norm_text = processor.apply_chat_template(norm_messages, tokenize=False, add_generation_prompt=True)
        safety_text = processor.apply_chat_template(safety_messages, tokenize=False, add_generation_prompt=True)
       
        norm_inputs = processor(
            text=[norm_text], images=[image], padding='max_length', truncation=True,max_length=1800, return_tensors="pt"
        )
        safety_inputs = processor(text=[safety_text], images=[image], padding='max_length', truncation=True,max_length=1800, return_tensors="pt")

        norm_loader.append(norm_inputs)
        safety_loader.append(safety_inputs)
    return norm_loader, safety_loader
