"""
DataGuard AI - Professional Fine-tuning Script
Model: Lily-1.5B (Based on Qwen2.5-1.5B)
Optimization: Unsloth + LoRA (r=32 for knowledge injection)

Run this script on a GPU machine (Colab, Lambda, Vast.ai etc.)
after generating the dataset with:
    python ml/insight_model/generate_training_data.py
"""

import os

try:
    from unsloth import FastLanguageModel # type: ignore
    import torch # type: ignore
    from trl import SFTTrainer # type: ignore
    from transformers import TrainingArguments # type: ignore
    from datasets import load_dataset # type: ignore
except ImportError:
    print("Dependencies not found. Install with: pip install 'unsloth[colab-new]' datasets trl")
    raise SystemExit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
DTYPE = None              # Auto-detect: fp16 on T4/V100, bf16 on Ampere+
LOAD_IN_4BIT = True       # 4-bit quantization for memory efficiency
LORA_RANK = 32            # Increased from 16 for better knowledge injection
LORA_ALPHA = 64           # 2x rank — standard practice for task-specific SFT
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../dataguard_lora_model")

# Number of training steps.
# Formula: (n_examples / batch_size / grad_accum) * n_epochs
# With 3000 samples, batch=2, grad_accum=4 → 375 steps/epoch → use 750 for 2 epochs
MAX_STEPS = 750

# ─── 1. Load Model & Tokenizer ─────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# ─── 2. Add LoRA Adapters ──────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,          # 0 is optimal for Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ─── 3. Prompt Template ────────────────────────────────────────────────────────
# EOS token is critical — without it the model won't learn when to stop.
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples: dict) -> dict:
    """
    Formats examples using the Alpaca instruction template.
    This template MUST match the format used in InsightEngine at inference time.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = (
            "Below is an instruction that describes a data quality analysis task. "
            "Write a structured professional response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
            f"{EOS_TOKEN}"  # Essential: teaches the model when to stop
        )
        texts.append(text)
    return {"text": texts}


# ─── 4. Load Dataset ──────────────────────────────────────────────────────────
# Path is relative to the script's directory for portability
data_path = os.path.join(os.path.dirname(__file__), "train_data.jsonl")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Training data not found at '{data_path}'.\n"
        "Generate it first by running:\n"
        "    python ml/insight_model/generate_training_data.py"
    )

dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Loaded {len(dataset)} training examples.")

# ─── 5. Training Arguments ────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # Effective batch size = 8
        warmup_steps=30,                    # ~4% of steps for warmup
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",         # Cosine decay > linear for SFT
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=250,                     # Checkpoint every 250 steps
        report_to="none",                   # Disable W&B by default
    ),
)

# ─── 6. Run Training ──────────────────────────────────────────────────────────
print("Starting DataGuard AI fine-tuning...")
trainer.train()

# ─── 7. Save Model ────────────────────────────────────────────────────────────
print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ─── 8. Export to GGUF for Ollama (uncomment when ready to deploy) ────────────
# print("Exporting to GGUF for Ollama...")
# model.save_pretrained_gguf(OUTPUT_DIR + "_gguf", tokenizer, quantization_method="q4_k_m")
# print("GGUF model saved. To load in Ollama:")
# print(f"  ollama create lily-dataguard -f {OUTPUT_DIR}_gguf/Modelfile")

print("Done! Fine-tuning complete.")
