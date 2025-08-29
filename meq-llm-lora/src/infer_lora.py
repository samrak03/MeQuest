import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "outputs/tinyllama-lora/lora_adapter"

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
m = PeftModel.from_pretrained(m, adapter_dir)
m = m.eval()

def chat(prompt, max_new_tokens=128):
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tok(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7)
    print(tok.decode(out[0], skip_special_tokens=True))

chat("WSL2에서 Docker와 GPU를 어떻게 연동해?")
