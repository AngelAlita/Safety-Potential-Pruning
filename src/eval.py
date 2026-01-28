import os
import argparse
import json
import torch
from tqdm import tqdm
from .LLamaGuard2 import LlamaGuard2
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    model = LlamaGuard2("cuda:0")
    result_file = f"{args.data_path}"
    print(f"Processing {result_file}")
    idx = 0
    count_safe = 0
    with open(result_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            idx += 1
            text = data['text']
            prompt = data['text']
            status = model.inference(text, prompt)
            if status == "safe":
                count_safe += 1
            data['status'] = status

    print(f'{count_safe * 1.0 / idx * 100}% of the samples are safe')

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, required=True)
    args = parse.parse_args()
    main(args)
