import argparse
import os
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from importlib.metadata import version

from src.utils import check_sparsity
from src.prune import prune_magnitude, prune_spp, prune_wanda, prune_sparsegpt


print('torch', version('torch'))
print('transformers',version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus', torch.cuda.device_count())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-7B-Instruct',help='VLM model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "SPP","sparsegpt"])
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--dataset', type=str, default='HOD', help='Dataset to be used for calibration.')
    args = parser.parse_args()

    # Setting seed for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split('/')[-1]
    print(f"loading VLM model {args.model}")
    if "Qwen2-VL" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model)
    elif "llava-v1.6" in model_name:
        processor = LlavaNextProcessor.from_pretrained(args.model)

        model = LlavaNextForConditionalGeneration.from_pretrained(args.model, 
                                                            torch_dtype=torch.float16, 
                                                            attn_implementation="flash_attention_2",
                                                            device_map="auto",
                                                            low_cpu_mem_usage=True) 
    elif "blip" in args.model:
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model,
                                                            torch_dtype=torch.float16,
                                                            device_map="auto",
                                                            low_cpu_mem_usage=True)
        processor = InstructBlipProcessor.from_pretrained(args.model)



    device = torch.device("cuda:0")
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "magnitude":
            prune_magnitude(args, model,processor, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda":
            prune_wanda(args, model,processor, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method =="SPP":
            prune_spp(args, model,processor, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model,processor, device, prune_n=prune_n, prune_m=prune_m)
        else:
            raise NotImplementedError

    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)


    if args.save_model:
        model.save_pretrained(args.save_model)
        processor.save_pretrained(args.save_model)

if __name__ == "__main__":
    main()
