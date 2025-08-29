import os, json, math, argparse
from dataclasses import dataclass
from typing import Dict, List
# import datasets
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def read_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_example(ex):
    # instruction / input / output -> single prompt
    if ex.get("input"):
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
    else:
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    return {"text": prompt}

def main(cfg):
    model_id = cfg["model_id"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    train_ds = load_dataset("json", data_files=cfg["train_file"])["train"]
    eval_ds = load_dataset("json", data_files=cfg["eval_file"])["train"]
    train_ds = train_ds.map(format_example)
    eval_ds  = eval_ds.map(format_example)

    max_len = cfg["train"]["max_seq_length"]
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(tok_fn, batched=True, remove_columns=eval_ds.column_names)

    # model load (8-bit or fp16)
    load_cfg = cfg["load"]
    load_kwargs = {}
    if load_cfg.get("load_in_8bit", False):
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **load_kwargs)

    if load_cfg.get("load_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # trainer
    t = cfg["train"]
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        # evaluation_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        fp16=cfg["load"].get("fp16", False),
        bf16=cfg["load"].get("bf16", False),
        report_to=[],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("eval metrics:", metrics)

    # 어댑터 저장(LoRA 가중치만)
    model.save_pretrained(os.path.join(out_dir, "lora_adapter"))
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/d/GitHub/MeQuest/meq-llm-lora/configs/run.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    main(cfg)
