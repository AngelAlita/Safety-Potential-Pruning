
import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    LlavaNextForConditionalGeneration, 
    LlavaNextProcessor,
    InstructBlipForConditionalGeneration, 
    InstructBlipProcessor
)
from qwen_vl_utils import process_vision_info

class VLMWrapper:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_name}")
        if "qwen2" in self.model_name.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        elif "llava" in self.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif "blip" in self.model_name.lower():
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.processor = InstructBlipProcessor.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def generate(self, image_path, text, max_new_tokens=256):
        if "qwen2" in self.model_name.lower():
            return self._generate_qwen2(image_path, text, max_new_tokens)
        elif "llava" in self.model_name.lower():
            return self._generate_llava(image_path, text, max_new_tokens)
        elif "blip" in self.model_name.lower():
            return self._generate_blip(image_path, text, max_new_tokens)
        else:
            raise NotImplementedError(f"Generation for {self.model_name} not implemented")

    def _generate_qwen2(self, image_path, text, max_new_tokens):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _generate_llava(self, image_path, text, max_new_tokens):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Using PIL Image
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)[0]
        return self.processor.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)

    def _generate_blip(self, image_path, text, max_new_tokens):
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
        )
        return self.processor.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()
