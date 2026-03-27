import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from retriever import retrieve
from peft import PeftModel

# model_path = "model/finetuned-llama-model"
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PEFT_PATH = "model/finetuned-llama-model"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
base_model = LlamaForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, PEFT_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def ask_rag(query):
    context_docs = retrieve(query)
    context = "\n".join(context_docs)

    prompt = f"""
    You are a helpful assistant.

    Use ONLY the context below to answer If the answer is not in the context, say "Not found in context".:
    
    Context:
    {context}

    Question: {query}
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # print(inputs.items())
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print('Done!')