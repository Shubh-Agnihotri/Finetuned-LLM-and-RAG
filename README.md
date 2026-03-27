# 🧠 Fine-Tuning TinyLlama with LoRA (PEFT)

This project demonstrates how to fine-tune the **TinyLlama-1.1B-Chat** model using **LoRA (Low-Rank Adaptation)** with Hugging Face libraries. It includes training, inference, and quick evaluation scripts.

---

## 🚀 Features

- Fine-tune TinyLlama using LoRA (PEFT)
- 4-bit quantization using BitsAndBytes (low memory usage)
- Custom dataset support (.jsonl format)
- Interactive inference script
- Quick evaluation using similarity score

---

## 📁 Project Structure

```
├── train.py                  # Training script
├── test.py                   # Inference & evaluation script
├── finetune.jsonl            # Dataset
├── model/
│   ├── finetuned-llama/
│   └── finetuned-llama-model/
```

---

## ⚙️ Installation

Install dependencies:

```bash
pip install torch transformers datasets peft bitsandbytes
```
---

## 📊 Dataset Format

Dataset must be in `.jsonl` format:

```json
{"instruction": "Your input text", "output": "Expected response"}
```

Example:

```json
{"instruction": "How to fix internet issues?", "output": "Restart router and check cables."}
```

---

## 🏋️ Training

Run:

```bash
python train.py
```

### Training Details

- Base Model: TinyLlama-1.1B-Chat
- LoRA Rank: 16
- LoRA Alpha: 32
- Target Modules: q_proj, v_proj
- Epochs: 3
- Learning Rate: 2e-4
- Quantization: 4-bit

Model is saved at:

```
model/finetuned-llama-model/
```

---

## 🤖 Inference

Run:

```bash
python test.py
```

Then enter your query:

```
Enter your query:
```

---

## 📈 Evaluation

The script performs quick evaluation using similarity score:

- Uses `difflib.SequenceMatcher`
- Score range: 0 to 1

Example:

```
Instruction: ...
Reference: ...
Prediction: ...
Similarity: 0.82
```

---

## 🧩 Workflow

### Training

1. Load TinyLlama model with 4-bit quantization  
2. Apply LoRA adapters  
3. Format data as:
   ```
   <s>instruction
   output</s>
   ```
4. Train using Hugging Face Trainer  

### Inference

1. Load base model + LoRA weights  
2. Format prompt:
   ```
   <s>instruction\n
   ```
3. Generate response using sampling  

---
