# import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
# from datasets import load_dataset
# import difflib

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# PEFT_PATH = "model/finetuned-llama-model"  # where trainer saved LoRA weights
# DATA_FILE = "finetune.jsonl"  # used for quick evaluation
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def load_model():
#     bnb_config = BitsAndBytesConfig(load_in_4bit=True)
#     tokenizer = LlamaTokenizer.from_pretrained(PEFT_PATH)
#     base_model = LlamaForCausalLM.from_pretrained(
#         PEFT_PATH,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )
#     model = PeftModel.from_pretrained(base_model, PEFT_PATH)
#     model.eval()
#     return tokenizer, model

# def make_prompt(instruction: str):
#     # follow same formatting used during training
#     return f"<s>{instruction}\n"

# def generate_once(tokenizer, model, instruction, max_new_tokens=128, temperature=0.7):
#     prompt = make_prompt(instruction)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     input_len = inputs["input_ids"].shape[-1]

#     with torch.no_grad():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=temperature,
#             top_k=50,
#             top_p=0.95,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )

#     # only decode the newly generated tokens (exclude the prompt)
#     gen_ids = out[0][input_len:]
#     pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#     return pred

# def quick_eval(n_examples=10):
#     tokenizer, model = load_model()
#     ds = load_dataset("json", data_files=DATA_FILE)["train"]
#     n = min(n_examples, len(ds))
#     results = []
#     for i in range(n):
#         ins = ds[i]["instruction"]
#         ref = ds[i].get("output", "")
#         pred = generate_once(tokenizer, model, ins)
#         score = difflib.SequenceMatcher(None, pred, ref).ratio()
#         results.append({"instruction": ins, "reference": ref, "prediction": pred, "sim": score})
#     return results

# if __name__ == "__main__":
#     # generate a single sample
#     tokenizer, model = load_model()
#     sample_instruction = "I am facing problems with my internet connection. How can I troubleshoot it?"
#     print("Instruction:", sample_instruction)
#     print("Prediction:", generate_once(tokenizer, model, sample_instruction))

#     # run quick evaluation on first 5 examples from finetune.jsonl
#     print("\nQuick evaluation on training file:")
#     eval_results = quick_eval(n_examples=5)
#     for r in eval_results:
#         print("-" * 40)
#         print("Instruction:", r["instruction"])
#         print("Reference:", r["reference"])
#         print("Prediction:", r["prediction"])
#         print("Similarity (0-1):", round(r["sim"], 3))

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import difflib

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PEFT_PATH = "model/finetuned-llama-model"  # where trainer saved LoRA weights
DATA_FILE = "finetune.jsonl"  # used for quick evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    # load the original base tokenizer/model, then apply the LoRA weights saved in PEFT_PATH
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    base_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, PEFT_PATH, device_map="auto")
    model.eval()
    return tokenizer, model

def make_prompt(instruction: str):
    # follow same formatting used during training
    return f"<s>{instruction}\n"

def generate_once(tokenizer, model, instruction, max_new_tokens=128, temperature=0.7):
    prompt = make_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # only decode the newly generated tokens (exclude the prompt)
    gen_ids = out[0][input_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred

def quick_eval(n_examples=10):
    tokenizer, model = load_model()
    ds = load_dataset("json", data_files=DATA_FILE)["train"]
    n = min(n_examples, len(ds))
    results = []
    for i in range(n):
        ins = ds[i]["instruction"]
        ref = ds[i].get("output", "")
        pred = generate_once(tokenizer, model, ins)
        score = difflib.SequenceMatcher(None, pred, ref).ratio()
        results.append({"instruction": ins, "reference": ref, "prediction": pred, "sim": score})
    return results

if __name__ == "__main__":
    # generate a single sample
    tokenizer, model = load_model()
    print("Enter your query:")
    sample_instruction = input()
    print("Instruction:", sample_instruction)
    print("Prediction:", generate_once(tokenizer, model, sample_instruction))

    # run quick evaluation on first 5 examples from finetune.jsonl
    print("\nQuick evaluation on training file:")
    eval_results = quick_eval(n_examples=5)
    for r in eval_results:
        print("-" * 40)
        print("Instruction:", r["instruction"])
        print("Reference:", r["reference"])
        print("Prediction:", r["prediction"])
        print("Similarity (0-1):", round(r["sim"], 3))