# 🛡️ Safety-Potential Pruning for Enhancing Safety Prompts Against VLM Jailbreaking Without Retraining



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)


This repository contains the official implementation for the TACL paper ["Safety-Potential Pruning for Enhancing Safety Prompts Against VLM Jailbreaking Without Retraining"](xxx).

![Framework](./assets/framework.png)

## 📄 Abstract
Safety prompts constitute an interpretable layer of defense against jailbreak attacks in vision–language models (VLMs); however, their efficacy is constrained by the models' latent structural responsiveness. We observe that such prompts consistently engage a sparse set of parameters that remain largely quiescent during benign use. This finding motivates the Safety Subnetwork Hypothesis: VLMs embed structurally distinct pathways capable of enforcing safety, but these pathways remain dormant without explicit stimulation. To expose and amplify these pathways, we introduce Safety-Potential Pruning, a one-shot pruning framework that amplifies safety-relevant activations by removing weights that are less responsive to safety prompts without additional retraining. Across three representative VLM architectures and three jailbreak benchmarks, our method reduces attack success rates by up to 22\% relative to prompting alone, all while maintaining strong benign performance. These findings frame pruning not only as a model compression technique, but as a structural intervention to emerge alignment-relevant subnets, offering a new path to robust jailbreak resistance.

## 🧸 Installation

1. **Create conda environment:**
   ```bash
   conda create -n SPP python=3.9 -y
   conda activate SPP
   ```
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```


## 📂 Data Preparation

Please download the datasets from the links provided in the [HOD](https://github.com/poori-nuna/HOD-Benchmark-Dataset), [MM-SafetyBench](https://github.com/isxinliu/MM-SafetyBench) and [FigStep](https://github.com/CryptoAILab/FigStep) then organize them in the `data` directory as follows:

```
data/
├── FigStep/
│   └── data/
│       ├── images/
│       │   └── SafeBench/
│       └── question/
│           └── safebench.csv
├── MM-SafetyBench/
│   ├── imgs/
│   └── processed_questions/
└── HOD/
    └── jpg/
    └── HOD.json
```

## 🚀 Usage
1. To prune a model using our proposed method with a sparsity ratio of 50%:
    ```bash
    python main.py 
        --model Qwen/Qwen2-VL-7B-Instruct 
        --prune_method SPP 
        --sparsity_ratio 0.5 
        --sparsity_type unstructured
        --save_model outputs/Qwen2-VL-7B-Instruct-SPP-50
    ```

2. To evaluate the safety of the pruned model:

    **Inference:**
    ```bash
    python src/inference_safety.py 
        --model outputs/Qwen2-VL-7B-Instruct-SPP-50 
        --dataset figstep 
        --data_path ./data 
        --safety_prompt True
        --save_path outputs/Qwen2-VL-7B-Instruct-SPP-50/figstep_safety.json
    ```
    **Evaluation:**
    We use Llama Guard2 to evaluate the safety of the pruned model.
    ```bash
    python src/eval.py --data_path outputs/Qwen2-VL-7B-Instruct-SPP-50/figstep_safety.json

    ```

3. To evaluate the utility of the pruned model, we recommend using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Since our pruned model maintains the original architecture, it can be seamlessly evaluated.

    **Installation:**
    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    pip install -e .
    ```

    **Evaluation:**
    You can evaluate the model on supported benchmarks. Ensure you configure the local path to your pruned model in `vlmeval/config.py`.

    ```bash
    python run.py --data MMMU_VAL --model Qwen2-VL-7B-Instruct --work-dir results/
    ```



## 🌻 Acknowledgments
This work builds upon the codebase of [Wanda](https://github.com/locuslab/wanda). We utilize and acknowledge the following benchmarks: [HOD](https://github.com/poori-nuna/HOD-Benchmark-Dataset), [MM-SafetyBench](https://github.com/isxinliu/MM-SafetyBench), [FigStep](https://github.com/CryptoAILab/FigStep), and [JailbreakV-28K](https://github.com/SaFo-Lab/JailBreakV_28K). We sincerely thank the authors for their valuable contributions to the community.


<!--   -->