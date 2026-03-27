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

# 🧠RAG Chatbot 

Now we build a **Retrieval-Augmented Generation (RAG) chatbot** on top of a **fine-tuned TinyLlama model** using **LoRA (PEFT)**.  
It combines **custom knowledge retrieval (ChromaDB)** with **LLM generation** for more accurate and context-aware responses.

---

## 🚀 Features

- Fine-tuned TinyLlama (LoRA / PEFT)
- Retrieval-Augmented Generation (RAG)
- ChromaDB as vector database
- Sentence Transformers for embeddings
- Context-aware chatbot
- Lightweight and efficient (4-bit compatible)

---

## 📁 Project Structure

```
├── train.py                  # Fine-tuning script
├── test.py                   # Inference script
├── retriever.py              # ChromaDB retrieval logic
├── finetune.jsonl            # Training dataset
├── model/
│   ├── finetuned-llama/
│   └── finetuned-llama-model/
```
---

## ⚙️ Installation

```bash
pip install torch transformers datasets peft bitsandbytes chromadb sentence-transformers
```
---

## 📊 Dataset Format

```json
{"instruction": "Your input", "output": "Expected output"}
```
---

## 🧠 RAG Pipeline

This part extends the fine-tuned model using a **RAG architecture**:

### Components

- **LLM** → Fine-tuned TinyLlama  
- **Vector Store** → ChromaDB  
- **Embeddings** → Sentence Transformers  
- **Retriever** → Custom `retrieve()` function  
---

## 🔄 RAG Flow

1. User query is received  
2. Query is converted into embeddings  
3. Relevant documents retrieved from ChromaDB  
4. Context is injected into prompt  
5. LLM generates answer using ONLY retrieved context  

---

---

## 📌 Key Advantages of RAG

- Reduces hallucination  
- Uses external knowledge  
- Keeps model lightweight  
- No need to retrain for new data  

---
