import os
import time
import heapq
import torch
from torch import nn

from .utils import find_layers, check_sparsity
from .layerwrapper import SPTWrappedGPT, WandaWrappedGPT
from .data import get_hod
from .sparsegpt import SparseGPT 

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llava-v1.6" in args.model:
        layers = model.language_model.model.layers
    elif "Qwen2-VL" in args.model:
        layers = model.model.layers
    elif "blip" in args.model:
        layers = model.language_model.model.layers
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    if "llava-v1.6" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], 4096), dtype=dtype, device=device) 
    elif "Qwen2-VL" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], model.config.hidden_size), dtype=dtype, device=device)
    elif "blip" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], 4096), dtype=dtype, device=device) 
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Cather(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Cather(layers[0])
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        try:
            model(**inputs)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_wanda(args, model, processor, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if 'llava-v1.6' in args.model or 'blip' in args.model:
        use_cache = True
    else:
        use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibadation data")
    norm_dataloader, safety_dataloader = get_hod(args, args.nsamples, args.seed, os.path.join('data', 'HOD', 'jpg'), processor)

    print("dataset loading comelete")
    with torch.no_grad():
        safety_inps, safety_outs, safety_attention_mask, safety_position_ids  = prepare_calibration_input(args, model, safety_dataloader, device)
    
    if "llava-v1.6" in args.model:
        layers = model.language_model.model.layers
    elif "Qwen2-VL" in args.model:
        layers = model.model.layers
    elif "blip" in args.model:
        layers = model.language_model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # for llava
        if f"language_model.model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"language_model.model.layers.{i}"]
            safety_inps, safety_outs, safety_position_ids = safety_inps.to(dev), safety_outs.to(dev), safety_position_ids.to(dev)
            if safety_attention_mask is not None:
                safety_attention_mask = safety_attention_mask.to(dev)

        # safety processing ##
        safety_wrapped_layers = {}
        for name in subset:
            safety_wrapped_layers[name] = WandaWrappedGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                safety_wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        safety_handles = []
        for name in safety_wrapped_layers:
            safety_handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in  range(args.nsamples):
            with torch.no_grad():
                safety_outs[j] = layer(safety_inps[j].unsqueeze(0), attention_mask=safety_attention_mask, position_ids=safety_position_ids)[0]

        for h in safety_handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(safety_wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii+ torch.topk(tmp,prune_n,dim=1,largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while( torch.abs(cur_sparsity - args.sparsity) > 0.001) and (alpha_hist[1] - alpha_hist[0]>0.001):
                        if cur_sparsity > args.sparsity:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity{cur_sparsity:.6f}")
                else:
                    ### unstructured pruning ### 
                    
                    indices = sort_res[1][:,:int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                safety_outs[j] = layer(safety_inps[j].unsqueeze(0), attention_mask=safety_attention_mask, position_ids=safety_position_ids)[0]
        safety_inps, safety_outs = safety_outs, safety_inps
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_spp(args,model, processor, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if 'llava-v1.6' or 'blip'in args.model:
        use_cache = True
    else:
        use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibadation data")
    norm_dataloader, safety_dataloader = get_hod(args, args.nsamples, args.seed, os.path.join('data', 'HOD', 'jpg'), processor)
    print("dataset loading comelete")
    with torch.no_grad():
        norm_inps, norm_outs, norm_attention_mask, norm_position_ids  = prepare_calibration_input(args, model, norm_dataloader, device)
        safety_inps, safety_outs, safety_attention_mask, safety_position_ids  = prepare_calibration_input(args, model, safety_dataloader, device)

    if "llava-v1.6" in args.model:
        layers = model.language_model.model.layers
    elif "Qwen2-VL" in args.model:
        layers = model.model.layers
    elif "blip" in args.model:
        layers = model.language_model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # for llava
        if f"language_model.model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"language_model.model.layers.{i}"]
            norm_inps, norm_outs, norm_position_ids = norm_inps.to(dev), norm_outs.to(dev),  norm_position_ids.to(dev)
            safety_inps, safety_outs, safety_position_ids = safety_inps.to(dev), safety_outs.to(dev), safety_position_ids.to(dev)
            if norm_attention_mask is not None:
                norm_attention_mask = norm_attention_mask.to(dev)
            if safety_attention_mask is not None:
                safety_attention_mask = safety_attention_mask.to(dev)

        # for qwen
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            norm_inps, norm_outs, norm_position_ids = norm_inps.to(dev), norm_outs.to(dev),  norm_position_ids.to(dev)
            safety_inps, safety_outs, safety_position_ids = safety_inps.to(dev), safety_outs.to(dev), safety_position_ids.to(dev)
            if norm_attention_mask is not None:
                norm_attention_mask = norm_attention_mask.to(dev)
            if safety_attention_mask is not None:
                safety_attention_mask = safety_attention_mask.to(dev)


        # norm processoing ##
        norm_wrapped_layers = {}
        for name in subset:
            norm_wrapped_layers[name] = SPTWrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                norm_wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        norm_handles = []
        for name in norm_wrapped_layers:
            norm_handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in  range(args.nsamples):
            with torch.no_grad():
                norm_outs[j] = layer(norm_inps[j].unsqueeze(0), attention_mask=norm_attention_mask, position_ids=norm_position_ids)[0]

        for h in norm_handles:
            h.remove()

        # safety processing ##
        safety_wrapped_layers = {}
        for name in subset:
            safety_wrapped_layers[name] = SPTWrappedGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                safety_wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        safety_handles = []
        for name in safety_wrapped_layers:
            safety_handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in  range(args.nsamples):
            with torch.no_grad():
                safety_outs[j] = layer(safety_inps[j].unsqueeze(0), attention_mask=safety_attention_mask, position_ids=safety_position_ids)[0]

        for h in safety_handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            l2 = torch.zeros((norm_wrapped_layers[name].columns),device="cpu")

            for j in range(safety_wrapped_layers[name].nsamples):
               l2 += safety_wrapped_layers[name].scaler_row[j] - norm_wrapped_layers[name].scaler_row[j] 
                
            l2 = l2 / safety_wrapped_layers[name].nsamples
            W_metric = torch.abs(subset[name].weight.data).to("cpu") * torch.sqrt(l2.reshape((1,-1))) 

            W_mask = (torch.zeros_like(W_metric) == 1)
            
        
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii+ torch.topk(tmp,prune_n,dim=1,largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while( torch.abs(cur_sparsity - args.sparsity) > 0.001) and (alpha_hist[1] - alpha_hist[0]>0.001):
                        if cur_sparsity > args.sparsity:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity{cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:,:int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                norm_outs[j] = layer(norm_inps[j].unsqueeze(0), attention_mask=norm_attention_mask, position_ids=norm_position_ids)[0]    
                safety_outs[j] = layer(safety_inps[j].unsqueeze(0), attention_mask=safety_attention_mask, position_ids=safety_position_ids)[0]

        norm_inps, norm_outs = norm_outs, norm_inps
        safety_inps, safety_outs = safety_outs, safety_inps
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
def prune_magnitude(args, model, processor, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if "llava-v1.6" in args.model:
        layers = model.language_model.model.layers
    elif "Qwen2-VL" in args.model:
        layers = model.model.layers
    elif "blip" in args.model:
        layers = model.language_model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric < thresh)

            W[W_mask] = 0

@torch.no_grad()
def prune_sparsegpt(args, model, processor, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    if 'llava-v1.6' or 'blip'in args.model:
        use_cache = True
    else:
        use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibadation data")
    _, dataloader = get_hod(args, args.nsamples, args.seed, os.path.join('data', 'HOD', 'jpg'), processor)
    print("dataset loading comelete")

    if "llava-v1.6" in args.model:
        layers = model.language_model.model.layers
    elif "Qwen2-VL" in args.model:
        layers = model.model.layers
    elif "blip" in args.model:
        layers = model.language_model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    
    if "llava-v1.6" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], 5120), dtype=dtype, device=device)
    elif "Qwen2-VL" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], model.config.hidden_size), dtype=dtype, device=device)
    elif "blip" in args.model:
        inps = torch.zeros((args.nsamples , dataloader[0]['input_ids'].shape[1], 4096), dtype=dtype, device=device)
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        try:
            model(**inputs)
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), None , position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
