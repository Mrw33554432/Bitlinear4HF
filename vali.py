import os
import time
import torch

from replace_hf import replace_linear_in_hf

os.environ['HF_HOME'] = r'D:\cache'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "results/checkpoint-37500"


def quick_test(model, tokenizer, prompt: str):
    # Encode the inputs
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate outputs
    outputs = model.generate(inputs, max_length=50)

    # Decode and print the outputs
    print(tokenizer.decode(outputs[0]))


torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
print(model)
# Replace Linear layers with BitLinear
replace_linear_in_hf(model, keep_param=True)
print(model)
start_time = time.time()
quick_test(model, tokenizer, prompt="Tom is the")
print(time.time() - start_time)
