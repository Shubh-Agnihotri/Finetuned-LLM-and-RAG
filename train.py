from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,    # or load_in_4bit=True
)

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load dataset
dataset = load_dataset("json", data_files="finetune.jsonl")

# def format_example(example):
#     prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n"
#     return {
#         "input_ids": tokenizer(prompt, return_tensors="pt", padding=False).input_ids[0],
#         "labels": tokenizer(example['output'], return_tensors="pt").input_ids[0]
#     }

def format(example):
    text = f"<s>{example['instruction']}\n{example['output']}</s>"
    tokens = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",   # padding ensures same length
    )
    tokens["labels"] = tokens["input_ids"].copy()  # labels = inputs
    return tokens

dataset = dataset.map(format, batched=False)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="model/finetuned-llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
trainer.save_model("model/finetuned-llama-model")
