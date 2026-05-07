"""
DataGuard AI - Professional Fine-tuning Script
Model: Lily-1.5B (Based on Qwen2.5-1.5B)
Optimization: Unsloth + LoRA
"""

try:
    from unsloth import FastLanguageModel # type: ignore
    import torch # type: ignore
    from trl import SFTTrainer # type: ignore
    from transformers import TrainingArguments # type: ignore
    from datasets import load_dataset # type: ignore
except ImportError:
    print("Dependencies not found. Install with: pip install unsloth [colab-new] datasets trl")
    exit()

# 1. Configuration
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", # Base model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Prepare Dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Alpaca-style prompt template
        text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="ml/insight_model/train_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Small number for demo/testing
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Run Training
print("Starting Fine-tuning...")
trainer.train()

# 7. Save and Export
print("Saving LoRA adapters...")
model.save_pretrained("dataguard_lora_model")
tokenizer.save_pretrained("dataguard_lora_model")

# Optional: Export to GGUF for Ollama
# model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

print("Done! You can now load this model into the platform.")
